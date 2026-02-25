%% Auto-plot + metrics (ONLY T0 >= 55 °C)
clc; clear; close all;

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');

%% ===== SCALING CONSTANTS =====
x_mean = 59.0676;
x_std  = 6.9122;

u_mean = 65.8447;
u_std  = 22.9062;

%% ===== SETTINGS =====
Q_cost = 10;
R_cost = 1;

temps = [45, 50, 55, 58, 60, 62, 66, 68];

% target used during experiments (mean reached with u_mean that day)
x_mean_target = 59.0676 + abs(59.0676 - 63.8084);

%% ===== OUTPUT FOLDER =====
if ~exist('figs','dir')
    mkdir('figs');
end

%% ===== STORAGE =====
RMSE_K = [];
RMSE_S = [];
IAE_K  = [];
IAE_S  = [];
OBJ_K  = [];
OBJ_S  = [];
T0_vec = [];

%% ===== MAIN LOOP =====
for k = 1:length(temps)

    T0 = temps(k);

    % === FILTER: ONLY T0 >= 55 °C ===
    if T0 < 55
        continue
    end

    fprintf('\n--- Processing start T = %d °C ---\n', T0);

    koop_file   = sprintf('steps/2611runtime_log_koop%d.mat', k);
    strejc_file = sprintf('steps/2611runtime_log_strejc%d.mat', k);

    %% ===== LOAD KOOPMAN =====
    S = load(koop_file);
    logK = S.log_data;

    T4_koop   = logK.T4(:);
    Pump_koop = logK.Pump2(:);

    %% ===== LOAD STREJC =====
    S = load(strejc_file);
    logS = S.log_data;

    T4_strejc   = logS.T4(:);
    Pump_strejc = logS.Pump2(:);

    %% ===== ALIGN LENGTHS =====
    L = min([length(T4_koop), length(T4_strejc), ...
             length(Pump_koop), length(Pump_strejc)]);
    T4_koop   = T4_koop(1:L);
    T4_strejc = T4_strejc(1:L);
    Pump_koop = Pump_koop(1:L);
    Pump_strejc = Pump_strejc(1:L);

    step = (0:L-1)';

    %% ===== METRICS =====
    target = x_mean_target;
    bias   = target - x_mean;

    % RMSE
    rmse_koop   = sqrt(mean((T4_koop   - target).^2));
    rmse_strejc = sqrt(mean((T4_strejc - target).^2));

    % IAE
    iae_koop   = sum(abs(T4_koop   - target));
    iae_strejc = sum(abs(T4_strejc - target));

    % Objective (scaled, MPC-consistent)
    obj_koop = sum( ...
        Q_cost*((T4_koop   - bias - x_mean)/x_std).^2 + ...
        R_cost*((Pump_koop - u_mean)/u_std).^2 );

    obj_strejc = sum( ...
        Q_cost*((T4_strejc - bias - x_mean)/x_std).^2 + ...
        R_cost*((Pump_strejc - u_mean)/u_std).^2 );

    % Store
    T0_vec = [T0_vec; T0];
    RMSE_K = [RMSE_K; rmse_koop];
    RMSE_S = [RMSE_S; rmse_strejc];
    IAE_K  = [IAE_K; iae_koop];
    IAE_S  = [IAE_S; iae_strejc];
    OBJ_K  = [OBJ_K; obj_koop];
    OBJ_S  = [OBJ_S; obj_strejc];

    %% ===== CLOSED-LOOP PLOT PER T0 =====
    fig = figure('Color','w','Position',[100 100 900 520]);
    tiledlayout(2,1,'TileSpacing','Compact','Padding','Compact');

    % --- OUTPUT ---
    nexttile;
    plot(step, T4_koop, 'm', 'LineWidth', 2); hold on;
    plot(step, T4_strejc, 'b--', 'LineWidth', 2);
    yline(target,'k','LineWidth',1.2);
    grid on; grid minor;
    ylabel('Outlet temperature ($^\circ$C)');
    title(sprintf('Closed-loop response (start %d$^\\circ$C)', T0));
    legend('Koopman MPC','Strejc MPC','Reference','Location','best');
    ylim([min([T4_koop;T4_strejc])-1.5, max([T4_koop;T4_strejc])+1.5]);

    % --- INPUT ---
    nexttile;
    plot(step, Pump_koop, 'm', 'LineWidth', 2); hold on;
    plot(step, Pump_strejc, 'b--', 'LineWidth', 2);
    grid on; grid minor;
    xlabel('Time step');
    ylabel('Pump speed (\%)');
    title('Control input');
    legend('Koopman MPC','Strejc MPC','Location','best');

    %saveas(fig, sprintf('figs/compare_cl_T0_%d.png', T0));
end

%% ===== METRICS vs INITIAL CONDITION (T0 >= 55 °C) =====
figM = figure('Color','w','Position',[100 100 980 360]);
tiledlayout(1,3,'TileSpacing','Compact','Padding','Compact');

% RMSE
nexttile;
plot(T0_vec, RMSE_K,'m-o','LineWidth',2); hold on;
plot(T0_vec, RMSE_S,'b-s','LineWidth',2);
grid on; grid minor;
xlabel('Initial temperature $T_0$ ($^\circ$C)');
ylabel('RMSE ($^\circ$C)');
title('RMSE');
legend('Koopman','Strejc','Location','best');

% IAE
nexttile;
plot(T0_vec, IAE_K,'m-o','LineWidth',2); hold on;
plot(T0_vec, IAE_S,'b-s','LineWidth',2);
grid on; grid minor;
xlabel('Initial temperature $T_0$ ($^\circ$C)');
ylabel('IAE');
title('IAE');
legend('Koopman','Strejc','Location','best');

% Objective
nexttile;
plot(T0_vec, OBJ_K,'m-o','LineWidth',2); hold on;
plot(T0_vec, OBJ_S,'b-s','LineWidth',2);
grid on; grid minor;
xlabel('Initial temperature $T_0$ ($^\circ$C)');
ylabel('Objective');
title('Objective value');
legend('Koopman','Strejc','Location','best');

%saveas(figM,'figs/metrics_vs_T0_from55.png');

