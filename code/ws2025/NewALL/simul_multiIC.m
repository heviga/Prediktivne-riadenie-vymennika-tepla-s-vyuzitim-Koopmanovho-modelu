%% cl_multiIC_koopman_strejc_baseline.m
% Multi-initial-condition CLOSED-LOOP simulations (baseline_inference as plant):
%   - Koopman MPC + KF
%   - Strejc  MPC + KF
% For each initial temperature T0:
%   - baseline is RESET separately for Koopman and Strejc
%   - 2-subplot figure like in simulation: T4 vs time, Pump vs time
%   - metrics per T0: RMSE, IAE, Objective (scaled consistent with MPC)
%   - summary table + mean + sum saved to .mat/.csv
%
% NOTE:
%   - baseline_inference works in scaled domain
%   - reference is 0 in scaled domain => physical reference is x_mean

clc; clear; close all;

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');

%% ===== USER SETTINGS =====
temps      = [45, 50, 55, 58, 60, 62, 66, 68];   % initial outlet temperatures [°C]
sim_length = 300;                                 % steps (Ts=1s)
Ts         = 1;

% MPC tuning (same for both)
N  = 20;
Qy = 10;
Qu = 1;

% constraints in PHYSICAL units
u_phys_min = 20;   % %
u_phys_max = 100;  % %
y_phys_min = 0;    % °C
y_phys_max = 70;   % °C

% Plot axis limits (your “normal” thesis scale)
y_lim = [58 64];
u_lim = [40 70];

%% ===== PATHS =====
addpath('../'); % readNPY

%% ===== PYTHON BASELINE SETUP =====
pyenv('Version', 'C:\Users\ivadu\AppData\Local\Programs\Python\Python39\python.exe');
py.sys.path().append('C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\ws2025\NewALL');
py.baseline_inference.init();

%% ===== LOAD SCALING DATA =====
load('train_data.mat');  % Ytrain, Utrain
load('test_data.mat');   % Ytest, Utest

Yall = [Ytrain(:); Ytest(:)];
Uall = [Utrain(:); Utest(:)];

x_mean = mean(Yall);
x_std  = std(Yall);
u_mean = mean(Uall);
u_std  = std(Uall);

% constraints in SCALED domain
umin = (u_phys_min - u_mean)/u_std;
umax = (u_phys_max - u_mean)/u_std;
ymin = (y_phys_min - x_mean)/x_std;
ymax = (y_phys_max - x_mean)/x_std;

% reference: 0 in scaled coordinates => x_mean in physical
y_ref_scaled = 0;
y_ref_phys   = x_mean;

%% ===== LOAD MODELS =====
% Koopman
A_k = double(readNPY('data/A_wC.npy'));
B_k = double(readNPY('data/B_wC.npy'));
C_k = double(readNPY('data/C_wC.npy'));
nx_k = size(A_k,1);

% Strejc (discrete, 1st order)
A_s = 0.98540172;
B_s = 0.01640409;
C_s = 1;
nx_s = 1;

%% ===== BUILD MPC OPTIMIZERS (ONCE) =====
% ---- Koopman MPC optimizer ----
uK = sdpvar(repmat(1,1,N), repmat(1,1,N));
xK = sdpvar(repmat(nx_k,1,N+1), repmat(1,1,N+1));
x0K = sdpvar(nx_k,1);

conK = [xK{1} == x0K];
objK = 0;
for k = 1:N
    conK = [conK, xK{k+1} == A_k*xK{k} + B_k*uK{k}];
    yk = C_k*xK{k};
    conK = [conK, umin <= uK{k} <= umax];
    conK = [conK, ymin <= yk <= ymax];
    objK = objK + Qy*(yk - y_ref_scaled)^2 + Qu*(uK{k})^2;
end
koopman_controller = optimizer(conK, objK, sdpsettings('solver','quadprog','verbose',0), x0K, uK{1});

% ---- Strejc MPC optimizer ----
uS = sdpvar(repmat(1,1,N), repmat(1,1,N));
xS = sdpvar(repmat(nx_s,1,N+1), repmat(1,1,N+1));
x0S = sdpvar(nx_s,1);

conS = [xS{1} == x0S];
objS = 0;
for k = 1:N
    conS = [conS, xS{k+1} == A_s*xS{k} + B_s*uS{k}];
    yk = C_s*xS{k};
    conS = [conS, umin <= uS{k} <= umax];
    conS = [conS, ymin <= yk <= ymax];
    objS = objS + Qy*(yk - y_ref_scaled)^2 + Qu*(uS{k})^2;
end
strejc_controller = optimizer(conS, objS, sdpsettings('solver','quadprog','verbose',0), x0S, uS{1});

%% ===== KALMAN FILTER SETTINGS =====
meas_noise_std = 0;

% Koopman KF
Q_KF = 0.5*eye(nx_k);
R_KF = 0.1;

% Strejc KF
Qk_s = 0.5;
Rk_s = 0.1;

%% ===== OUTPUT FOLDERS =====
if ~exist('figs','dir'), mkdir('figs'); end
if ~exist('figs/pdf','dir'), mkdir('figs/pdf'); end
if ~exist('figs/fig','dir'), mkdir('figs/fig'); end

%% ===== STORAGE FOR METRICS =====
metrics_rows = table('Size',[0 7], ...
    'VariableTypes', {'double','double','double','double','double','double','double'}, ...
    'VariableNames', {'T0','RMSE_K','IAE_K','Obj_K','RMSE_S','IAE_S','Obj_S'});

%% ===== MAIN LOOP OVER INITIAL CONDITIONS =====
for i = 1:length(temps)

    T0_phys = temps(i);
    fprintf('\n=== IC %d/%d: T0 = %d °C ===\n', i, length(temps), T0_phys);

    % initial output in scaled domain
    y0 = (T0_phys - x_mean)/x_std;

    %% ===== KOOPMAN CLOSED-LOOP (baseline plant) =====
    x0_k = pinv(C_k) * y0;

    x_est_k  = zeros(nx_k, sim_length+1);
    Pk       = eye(nx_k);
    u_k      = zeros(sim_length,1);
    y_true_k = zeros(sim_length+1,1);
    y_meas_k = zeros(sim_length+1,1);

    x_est_k(:,1) = x0_k;
    y_true_k(1)  = y0;
    y_meas_k(1)  = y_true_k(1) + meas_noise_std*randn;

    % RESET baseline state for Koopman run
    py.baseline_inference.get_x(y_true_k(1));

    for t = 1:sim_length
        u_k(t) = koopman_controller{x_est_k(:,t)};

        y_next = py.baseline_inference.y_plus(u_k(t));
        y_true_k(t+1) = double(y_next.item());

        y_meas_k(t+1) = y_true_k(t+1) + meas_noise_std*randn;

        x_pred = A_k*x_est_k(:,t) + B_k*u_k(t);
        P_pred = A_k*Pk*A_k' + Q_KF;

        Sinnov = C_k*P_pred*C_k' + R_KF;
        Kk = (P_pred*C_k')/Sinnov;

        x_est_k(:,t+1) = x_pred + Kk*(y_meas_k(t+1) - C_k*x_pred);
        Pk = (eye(nx_k) - Kk*C_k)*P_pred;
        Pk = (Pk + Pk')/2;
    end

    yK_phys = y_true_k*x_std + x_mean;
    uK_phys = u_k*u_std + u_mean;

    eK = yK_phys(1:end-1) - y_ref_phys;
    rmseK = sqrt(mean(eK.^2));
    iaeK  = sum(abs(eK));

    objK = sum( Qy*((yK_phys(1:end-1) - y_ref_phys)/x_std).^2 + ...
                Qu*((uK_phys - u_mean)/u_std).^2 );

    %% ===== STREJC CLOSED-LOOP (baseline plant) =====
    x_est_s  = zeros(sim_length+1,1);
    Ps       = 1;
    u_s      = zeros(sim_length,1);
    y_true_s = zeros(sim_length+1,1);
    y_meas_s = zeros(sim_length+1,1);

    x_est_s(1)  = y0;
    y_true_s(1) = y0;
    y_meas_s(1) = y_true_s(1) + meas_noise_std*randn;

    % RESET baseline state for Strejc run
    py.baseline_inference.get_x(y_true_s(1));

    for t = 1:sim_length
        u_s(t) = strejc_controller{x_est_s(t)};

        y_next = py.baseline_inference.y_plus(u_s(t));
        y_true_s(t+1) = double(y_next.item());

        y_meas_s(t+1) = y_true_s(t+1) + meas_noise_std*randn;

        x_pred = A_s*x_est_s(t) + B_s*u_s(t);
        P_pred = A_s*Ps*A_s' + Qk_s;

        Sinnov = C_s*P_pred*C_s' + Rk_s;
        Ks = (P_pred*C_s')/Sinnov;

        x_est_s(t+1) = x_pred + Ks*(y_meas_s(t+1) - C_s*x_pred);
        Ps = (1 - Ks*C_s)*P_pred;
        Ps = max(Ps, 1e-6);
    end

    yS_phys = y_true_s*x_std + x_mean;
    uS_phys = u_s*u_std + u_mean;

    eS = yS_phys(1:end-1) - y_ref_phys;
    rmseS = sqrt(mean(eS.^2));
    iaeS  = sum(abs(eS));

    objS = sum( Qy*((yS_phys(1:end-1) - y_ref_phys)/x_std).^2 + ...
                Qu*((uS_phys - u_mean)/u_std).^2 );

    %% ===== SAVE METRICS ROW =====
    metrics_rows = [metrics_rows; ...
        table(T0_phys, rmseK, iaeK, objK, rmseS, iaeS, objS, ...
        'VariableNames', {'T0','RMSE_K','IAE_K','Obj_K','RMSE_S','IAE_S','Obj_S'})];

    fprintf('T0=%d: Koop RMSE=%.3f IAE=%.1f Obj=%.2f | Strejc RMSE=%.3f IAE=%.1f Obj=%.2f\n', ...
        T0_phys, rmseK, iaeK, objK, rmseS, iaeS, objS);

    %% ===== PLOT (2 SUBPLOTS) + SAVE =====
    yK_phys = yK_phys(:); yS_phys = yS_phys(:);
    uK_phys = uK_phys(:); uS_phys = uS_phys(:);

    tY = (0:length(yK_phys)-1)';  % 0..sim_length
    tU = (0:length(uK_phys)-1)';  % 0..sim_length-1

    fig = figure('Color','w','Position',[100 100 900 520]);
    tiledlayout(2,1,'TileSpacing','Compact','Padding','Compact');

    % OUTPUT
    nexttile;
    plot(tY, yK_phys, 'm',   'LineWidth', 2); hold on;
    plot(tY, yS_phys, 'b--', 'LineWidth', 2);
    yline(y_ref_phys, 'k-', 'LineWidth', 1.2);
    grid on; grid minor;
    ylabel('Outlet temperature ($^\circ$C)');
    title(sprintf('Closed-loop response (start %d$^\\circ$C)', T0_phys));
    legend('Koopman MPC','Strejc MPC','Steady-state','Location','best');
    ylim(y_lim);
    xlim([0 sim_length]);

    % INPUT
    nexttile;
    plot(tU, uK_phys, 'm',   'LineWidth', 2); hold on;
    plot(tU, uS_phys, 'b--', 'LineWidth', 2);
    grid on; grid minor;
    xlabel('Time step');
    ylabel('Pump speed (\%)');
    title('Control input');
    legend('Koopman MPC','Strejc MPC','Location','best');
    ylim(u_lim);
    xlim([0 sim_length-1]);

    % ---- SAVE: PNG (300 dpi), PDF (vector), FIG ----
    fname = sprintf('sim_cl_compare_T0_%d', T0_phys);

    exportgraphics(fig, fullfile('figs', [fname '.png']), 'Resolution', 300);
    exportgraphics(fig, fullfile('figs','pdf', [fname '.pdf']), 'ContentType', 'vector');
    savefig(fig, fullfile('figs','fig', [fname '.fig']));

    close(fig);
end

%% ===== SUMMARY: MEAN + SUM =====
mean_row = table( ...
    -1, ...
    mean(metrics_rows.RMSE_K), mean(metrics_rows.IAE_K), mean(metrics_rows.Obj_K), ...
    mean(metrics_rows.RMSE_S), mean(metrics_rows.IAE_S), mean(metrics_rows.Obj_S), ...
    'VariableNames', metrics_rows.Properties.VariableNames);

sum_row = table( ...
    -2, ...
    sum(metrics_rows.RMSE_K), sum(metrics_rows.IAE_K), sum(metrics_rows.Obj_K), ...
    sum(metrics_rows.RMSE_S), sum(metrics_rows.IAE_S), sum(metrics_rows.Obj_S), ...
    'VariableNames', metrics_rows.Properties.VariableNames);

metrics_full = [metrics_rows; mean_row; sum_row];

disp('===== PERFORMANCE METRICS PER INITIAL TEMPERATURE =====');
disp(metrics_rows);

disp('===== MEAN (T0=-1) AND SUM (T0=-2) =====');
disp(metrics_full(end-1:end,:));

%% ===== SAVE TABLES =====
% save('results_multiIC_koopman_strejc.mat', ...
%     'metrics_rows','metrics_full','temps', ...
%     'x_mean','x_std','u_mean','u_std','sim_length','N','Qy','Qu');

writetable(metrics_full, 'results_multiIC_koopman_strejc.csv');

fprintf('\nSaved:\n  results_multiIC_koopman_strejc.mat\n  results_multiIC_koopman_strejc.csv\n');
fprintf('Figures saved into:\n  figs/*.png\n  figs/pdf/*.pdf\n  figs/fig/*.fig\n\n');



%% ===== SUMMARY PLOTS: RMSE / IAE / Objective =====
% uses metrics_rows (per initial condition only)

% make sure output folders exist (already created above, but safe)
if ~exist('figs','dir'), mkdir('figs'); end
if ~exist(fullfile('figs','pdf'),'dir'), mkdir(fullfile('figs','pdf')); end
if ~exist(fullfile('figs','fig'),'dir'), mkdir(fullfile('figs','fig')); end

T0 = metrics_rows.T0;

figM = figure('Color','w','Position',[100 100 1050 360]);
tiledlayout(1,3,'TileSpacing','Compact','Padding','Compact');

% --- RMSE ---
nexttile;
plot(T0, metrics_rows.RMSE_K, 'm-o', 'LineWidth', 1.6, 'MarkerSize', 6); hold on;
plot(T0, metrics_rows.RMSE_S, 'b-s','LineWidth', 1.6, 'MarkerSize', 6);
grid on; grid minor; box on;
xlabel('Initial temperature $T_0$ ($^\circ$C)');
ylabel('RMSE ($^\circ$C)');
title('Tracking RMSE');
legend('Koopman MPC','Strejc MPC','Location','northwest');

% --- IAE ---
nexttile;
plot(T0, metrics_rows.IAE_K, 'm-o', 'LineWidth', 1.6, 'MarkerSize', 6); hold on;
plot(T0, metrics_rows.IAE_S, 'b-s','LineWidth', 1.6, 'MarkerSize', 6);
grid on; grid minor; box on;
xlabel('Initial temperature $T_0$ ($^\circ$C)');
ylabel('IAE');
title('IAE');
legend('Koopman MPC','Strejc MPC','Location','northwest');

% --- Objective (closed-loop) ---
nexttile;
plot(T0, metrics_rows.Obj_K, 'm-o', 'LineWidth', 1.6, 'MarkerSize', 6); hold on;
plot(T0, metrics_rows.Obj_S, 'b-s','LineWidth', 1.6, 'MarkerSize', 6);
grid on; grid minor; box on;
xlabel('Initial temperature $T_0$ ($^\circ$C)');
ylabel('$J_{\mathrm{CL}}$');
title('Ojective value');
legend('Koopman MPC','Strejc MPC','Location','northwest');

% save summary figure
exportgraphics(figM, fullfile('figs','simul_metrics_summary.png'), 'Resolution', 300);
exportgraphics(figM, fullfile('figs','pdf','simul_metrics_summary.pdf'), 'ContentType', 'vector');
savefig(figM, fullfile('figs','fig','simul_metrics_summary.fig'));
