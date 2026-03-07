clc; clear; close all;
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');

project_root = 'C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code';
addpath(genpath(project_root));
addpath('../');

%% ===== PYTHON BASELINE =====
pyenv('Version', 'C:\Users\ivadu\AppData\Local\Programs\Python\Python39\python.exe');
py.sys.path().append( ...
    'C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\LS2026\closed loop simulations');
py.baseline_inference.init();

%% ===== LOAD MODELS =====
% Koopman model
A_k = double(readNPY('data/A_wC_all.npy'));
B_k = double(readNPY('data/B_wC_all.npy'));
C_k = double(readNPY('data/C_wC_all.npy'));
D_k = 0;

% Strejc model
A_s = 0.97701252;
B_s = 0.03150894;
C_s = 1;
D_s = 0;

%% ===== LOAD SCALING DATA =====
load('train_data_ident.mat');
load('test_data_ident.mat');

Ytrain = Ytrain(:);
Utrain = Utrain(:);
Ytest = Ytest(:);
Utest = Utest(:);

Yall = [Ytrain; Ytest];
Uall = [Utrain; Utest];

x_mean = mean(Yall);
x_std  = std(Yall);
u_mean = mean(Uall);
u_std  = std(Uall);

%% ===== MPC PARAMETERS =====
Qy = 10;
Qu = 1;
N = 40;
sim_length = 150;

umin = (20 - u_mean) / u_std;
umax = (100 - u_mean) / u_std;
ymin = (0 - x_mean) / x_std;
ymax = (70 - x_mean) / x_std;

%% ===== KOOPMAN MPC CONTROLLER =====
nx_k = size(A_k,1);

u_k = sdpvar(repmat(1,1,N), repmat(1,1,N));
x_k = sdpvar(repmat(nx_k,1,N+1), repmat(1,1,N+1));
x0_k = sdpvar(nx_k,1);

constraints = [x_k{1} == x0_k];
objective = 0;

for k = 1:N
    constraints = [constraints, x_k{k+1} == A_k * x_k{k} + B_k * u_k{k}];
    yk = C_k * x_k{k};
    constraints = [constraints, umin <= u_k{k} <= umax];
    constraints = [constraints, ymin <= yk <= ymax];
    objective = objective + Qy * (yk)^2 + Qu * u_k{k}^2;
end

koopman_controller = optimizer(constraints, objective, ...
    sdpsettings('solver','quadprog'), x0_k, u_k{1});

%% ===== STREJC MPC CONTROLLER =====
nx_s = 1;

u_s = sdpvar(repmat(1,1,N), repmat(1,1,N));
x_s = sdpvar(repmat(1,1,N+1), repmat(1,1,N+1));
x0_s = sdpvar(1,1);

constraints = [x_s{1} == x0_s];
objective = 0;

for k = 1:N
    constraints = [constraints, x_s{k+1} == A_s * x_s{k} + B_s * u_s{k}];
    y_k = C_s * x_s{k};
    constraints = [constraints, umin <= u_s{k} <= umax];
    constraints = [constraints, ymin <= y_k <= ymax];
    objective = objective + Qy * (y_k)^2 + Qu * u_s{k}^2;
end

strejc_controller = optimizer(constraints, objective, ...
    sdpsettings('solver','quadprog'), x0_s, u_s{1});

%% ===== KALMAN FILTER PARAMETERS =====
% Koopman KF
Q_KF = 0.5 * eye(nx_k);
R_KF = 0.1;
P_k = eye(nx_k);
meas_noise_std = 0;

% Strejc KF
Q_kalman = 0.5;
R_kalman = 0.1;
P_s = 1;

%% ===== INITIAL CONDITIONS =====
y0_vals = [45 50 55 58 60 62 66 68];
y0_vals_scaled = (y0_vals - x_mean) / x_std;

results_koopman = [];
results_strejc = [];

koopman_traj = cell(length(y0_vals),1);
strejc_traj  = cell(length(y0_vals),1);

%% ===== RUN TESTS =====
for i = 1:length(y0_vals_scaled)
    fprintf('Simulation %d / %d\n', i, length(y0_vals_scaled));

    y0_scaled = y0_vals_scaled(i);
    x0_k_init = pinv(C_k) * y0_scaled;

    %% --- KOOPMAN WITH BASELINE INTEGRATION ---
    xk = zeros(nx_k, sim_length+1);
    yk = zeros(1, sim_length+1);
    uk = zeros(1, sim_length);
    x_est_k = zeros(nx_k, sim_length+1);
    y_meas_k = zeros(1, sim_length+1);
    y_true_k = zeros(1, sim_length+1);

    P_k = eye(nx_k);

    xk(:,1) = x0_k_init;
    x_est_k(:,1) = x0_k_init;
    yk(:,1) = C_k * x0_k_init;
    y_true_k(:,1) = yk(:,1);
    y_meas_k(:,1) = y_true_k(:,1) + meas_noise_std * randn(1,1);

    for t = 1:sim_length
        try
            uk(:,t) = koopman_controller{x_est_k(:,t)};
            if isnan(uk(:,t))
                fprintf('Warning: NaN koopman control input at t=%d, IC=%d\n', t, i);
                uk(:,t) = 0;
            end
        catch ME
            fprintf('Error in Koopman controller at t=%d, IC=%d: %s\n', t, i, ME.message);
            uk(:,t) = 0;
        end

        if t == 1
            py.baseline_inference.get_x(y_true_k(:,t));
        end

        try
            y_baseline = py.baseline_inference.y_plus(uk(:,t));
            y_true_k(:,t+1) = double(y_baseline.item());

            if isnan(y_true_k(:,t+1))
                fprintf('Warning: NaN baseline output at t=%d, IC=%d\n', t, i);
                y_true_k(:,t+1) = y_true_k(:,t);
            end
        catch ME
            fprintf('Error in baseline inference at t=%d, IC=%d: %s\n', t, i, ME.message);
            y_true_k(:,t+1) = y_true_k(:,t);
        end

        y_meas_k(:,t+1) = y_true_k(:,t+1) + meas_noise_std * randn(1,1);

        x_pred = A_k * x_est_k(:,t) + B_k * uk(:,t);
        P_pred = A_k * P_k * A_k' + Q_KF;

        S = C_k * P_pred * C_k' + R_KF;
        K_gain = (P_pred * C_k') / S;
        x_est_k(:,t+1) = x_pred + K_gain * (y_meas_k(:,t+1) - C_k * x_pred);
        P_k = (eye(nx_k) - K_gain * C_k) * P_pred;
        P_k = (P_k + P_k') / 2;

        xk(:,t+1) = A_k * xk(:,t) + B_k * uk(:,t);
        yk(:,t+1) = C_k * xk(:,t+1);
    end

    yk_desc = y_true_k * x_std + x_mean;
    uk_desc = uk * u_std + u_mean;
    e_k = yk_desc(1:end-1);

    u_sum_koop = sum(abs(uk_desc));
    y_sum_koop = sum(abs(e_k));
    e_sum_koop_y = sum(abs(e_k - x_mean));
    e_sum_koop_u = sum(abs(uk_desc - u_mean));
    rmse_koop = sqrt(mean((e_k - x_mean).^2));
    obj_koop = sum(Qy * (y_true_k(1:end-1)).^2 + Qu * (uk).^2);

    koopman_metrics = [u_sum_koop, y_sum_koop, e_sum_koop_y, e_sum_koop_u, rmse_koop, obj_koop];
    results_koopman = [results_koopman; koopman_metrics];

    %% --- STREJC WITH BASELINE INTEGRATION ---
    xs = zeros(1, sim_length+1);
    ys = zeros(1, sim_length+1);
    us = zeros(1, sim_length);
    x_est_s = zeros(1, sim_length+1);
    y_meas_s = zeros(1, sim_length+1);
    y_true_s = zeros(1, sim_length+1);

    P_s = 1;

    xs(:,1) = y0_scaled;
    x_est_s(:,1) = y0_scaled;
    ys(:,1) = C_s * xs(:,1);
    y_true_s(:,1) = ys(:,1);
    y_meas_s(:,1) = y_true_s(:,1) + meas_noise_std * randn(1,1);

    for t = 1:sim_length
        try
            us(:,t) = strejc_controller{x_est_s(:,t)};
            if isnan(us(:,t))
                fprintf('Warning: NaN strejc control input at t=%d, IC=%d\n', t, i);
                us(:,t) = 0;
            end
        catch ME
            fprintf('Error in Strejc controller at t=%d, IC=%d: %s\n', t, i, ME.message);
            us(:,t) = 0;
        end

        if t == 1
            py.baseline_inference.get_x(y_true_s(:,t));
        end

        try
            y_baseline = py.baseline_inference.y_plus(us(:,t));
            y_true_s(:,t+1) = double(y_baseline.item());

            if isnan(y_true_s(:,t+1))
                fprintf('Warning: NaN baseline output at t=%d, IC=%d\n', t, i);
                y_true_s(:,t+1) = y_true_s(:,t);
            end
        catch ME
            fprintf('Error in baseline inference at t=%d, IC=%d: %s\n', t, i, ME.message);
            y_true_s(:,t+1) = y_true_s(:,t);
        end

        y_meas_s(:,t+1) = y_true_s(:,t+1) + meas_noise_std * randn(1,1);

        x_pred = A_s * x_est_s(:,t) + B_s * us(:,t);
        P_pred = A_s * P_s * A_s' + Q_kalman;

        S = C_s * P_pred * C_s' + R_kalman;
        if abs(S) < 1e-10
            K_gain = 0;
        else
            K_gain = (P_pred * C_s') / S;
        end

        x_est_s(:,t+1) = x_pred + K_gain * (y_meas_s(:,t+1) - C_s * x_pred);
        P_s = (1 - K_gain * C_s) * P_pred;
        P_s = max(P_s, 1e-6);

        xs(:,t+1) = A_s * xs(:,t) + B_s * us(:,t);
        ys(:,t+1) = C_s * xs(:,t+1);
    end

    ys_desc = y_true_s * x_std + x_mean;
    us_desc = us * u_std + u_mean;
    e_s = ys_desc(1:end-1);

    if any(isnan(us_desc)) || any(isnan(ys_desc))
        fprintf('Warning: NaN values detected in Strejc results for IC %d\n', i);
        us_desc(isnan(us_desc)) = 0;
        ys_desc(isnan(ys_desc)) = x_mean;
        e_s = ys_desc(1:end-1);
    end

    u_sum_strejc = sum(abs(us_desc));
    y_sum_strejc = sum(abs(e_s));
    e_sum_strejc_y = sum(abs(e_s - x_mean));
    e_sum_strejc_u = sum(abs(us_desc - u_mean));
    rmse_strejc = sqrt(mean((e_s - x_mean).^2));
    obj_strejc = sum(Qy * (y_true_s(1:end-1)).^2 + Qu * (us).^2);

    if any(isnan([u_sum_strejc, y_sum_strejc, e_sum_strejc_y, e_sum_strejc_u, rmse_strejc, obj_strejc]))
        fprintf('Warning: NaN metrics for Strejc IC %d, setting to zero\n', i);
        u_sum_strejc = 0; y_sum_strejc = 0; e_sum_strejc_y = 0;
        e_sum_strejc_u = 0; rmse_strejc = 0; obj_strejc = 0;
    end

    strejc_metrics = [u_sum_strejc, y_sum_strejc, e_sum_strejc_y, e_sum_strejc_u, rmse_strejc, obj_strejc];
    results_strejc = [results_strejc; strejc_metrics];

    %% --- STORE TRAJECTORIES ---
    koopman_traj{i}.y = yk_desc(:);
    koopman_traj{i}.u = uk_desc(:);
    koopman_traj{i}.y_scaled = y_true_k(:);
    koopman_traj{i}.u_scaled = uk(:);

    strejc_traj{i}.y = ys_desc(:);
    strejc_traj{i}.u = us_desc(:);
    strejc_traj{i}.y_scaled = y_true_s(:);
    strejc_traj{i}.u_scaled = us(:);

    %% --- INDIVIDUAL PLOTS ---
    tY = 0:sim_length;
    tU = 0:sim_length-1;

    figure('Name', sprintf('IC_%02d_y0_%.2f', i, y0_vals(i)), ...
           'Color','w','Position',[100 100 900 520]);

    tiledlayout(2,1,'TileSpacing','Compact','Padding','Compact');

    nexttile;
    plot(tY, yk_desc, 'm', 'LineWidth', 2); hold on;
    plot(tY, ys_desc, 'b--', 'LineWidth', 2);
    yline(x_mean, 'k', 'LineWidth', 1.2);
    grid on; grid minor;
    ylabel('Outlet temperature ($^\circ$C)');
    title(sprintf('Closed-loop response for $y_0 = %.2f\\,^\\circ$C', y0_vals(i)));
    legend('Koopman MPC','Strejc MPC','Steady-state','Location','best');

    nexttile;
    plot(tU, uk_desc, 'm', 'LineWidth', 2); hold on;
    plot(tU, us_desc, 'b--', 'LineWidth', 2);
    grid on; grid minor;
    xlabel('Time step');
    ylabel('Pump speed (\%)');
    legend('Koopman MPC','Strejc MPC','Location','best');
end

%% ===== RESULTS PRINT =====
fprintf('\n--- Koopman Results (sum|u| sum|y| sum|e|(y) sum|e|(u) RMSE Obj) ---\n');
for i = 1:size(results_koopman,1)
    fprintf('IC #%2d (y0=%.2f°C):  sum|u|=%.2f  sum|y|=%.2f  sum|e|(y)=%.2f  sum|e|(u)=%.2f  RMSE=%.2f  Obj=%.2f\n', ...
        i, y0_vals(i), results_koopman(i,1), results_koopman(i,2), ...
        results_koopman(i,3), results_koopman(i,4), results_koopman(i,5), results_koopman(i,6));
end

fprintf('\n--- Strejc Results (sum|u| sum|y| sum|e|(y) sum|e|(u) RMSE Obj) ---\n');
for i = 1:size(results_strejc,1)
    fprintf('IC #%2d (y0=%.2f°C):  sum|u|=%.2f  sum|y|=%.2f  sum|e|(y)=%.2f  sum|e|(u)=%.2f  RMSE=%.2f  Obj=%.2f\n', ...
        i, y0_vals(i), results_strejc(i,1), results_strejc(i,2), ...
        results_strejc(i,3), results_strejc(i,4), results_strejc(i,5), results_strejc(i,6));
end

%% ===== SUMMARY METRICS PLOT =====
metric_names = {'RMSE', 'Objective function'};
metric_idx = [5, 6];

figure('Name','Summary metrics','Position',[100 100 900 400],'Color','w');
for i = 1:2
    subplot(1,2,i)
    plot(y0_vals, results_koopman(:,metric_idx(i)), 'm-o','LineWidth',2); hold on;
    plot(y0_vals, results_strejc(:,metric_idx(i)), 'b-s','LineWidth',2);
    xlabel('Initial condition $y_0$ ($^\circ$C)');
    if i == 1
        ylabel('RMSE ($^\circ$C)');
    else
        ylabel('Objective function');
    end
    title(metric_names{i});
    legend('Koopman','Strejc','Location','best');
    grid on; grid minor;
end

%% ===== SUMMARY OF METRICS =====
metrics_labels = {'Sum |u|', 'Sum |y|', 'Sum e (y)', 'Sum e (u)', 'RMSE', 'Objective'};

koopman_mean   = mean(results_koopman);
strejc_mean    = mean(results_strejc);
koopman_median = median(results_koopman);
strejc_median  = median(results_strejc);
koopman_std    = std(results_koopman);
strejc_std     = std(results_strejc);

fprintf('\n========== Koopman vs Strejc - Statistics Summary ==========\n');
for i = 1:length(metrics_labels)
    fprintf('\nMetric: %s\n', metrics_labels{i});
    fprintf('  Koopman -> Mean: %.2f, Median: %.2f, Std: %.2f\n', ...
        koopman_mean(i), koopman_median(i), koopman_std(i));
    fprintf('  Strejc  -> Mean: %.2f, Median: %.2f, Std: %.2f\n', ...
        strejc_mean(i), strejc_median(i), strejc_std(i));
end

%% ===== OPTIONAL SAVE =====
% save('results_fixedIC_ws2025.mat', ...
%     'results_koopman', 'results_strejc', 'y0_vals', ...
%     'koopman_traj', 'strejc_traj');