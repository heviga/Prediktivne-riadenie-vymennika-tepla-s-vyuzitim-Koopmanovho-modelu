%% Auto-plot + metrics for ALL start temperatures (each T0 -> 1 figure like simulation)
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
time_format = 'yyyy-MM-dd HH:mm:ss.SSS';

Q_cost = 10;
R_cost = 1;

temps = [45, 50, 55, 58, 60, 62, 66, 68];
x_mean_target = 59.0676 + abs(59.0676 - 63.8084);  % your "target"

%% output folder
if ~exist('figs','dir')
    mkdir('figs');
end

%% ===== GLOBAL ACCUMULATORS =====
sum_obj_koop_scaled_all   = 0;
sum_obj_strejc_scaled_all = 0;

sum_iae_koop_all   = 0;
sum_iae_strejc_all = 0;

sum_rmse_koop_all   = 0;
sum_rmse_strejc_all = 0;

%% optional: store per-T0 summary into one table
all_rows = [];

%% ===== MAIN LOOP =====
for k = 1:length(temps)

    T0 = temps(k);
    fprintf('\n--- Processing start T = %d °C ---\n', T0);

    koop_file   = sprintf('steps/2611runtime_log_koop%d.mat', k);
    strejc_file = sprintf('steps/2611runtime_log_strejc%d.mat', k);

    %% ===== LOAD KOOPMAN =====
    S = load(koop_file);
    log_data = S.log_data;

    t = datetime(string(log_data.timestamp), 'InputFormat', time_format);
    time_koop = seconds(t - t(1));              % seconds (for "time step-ish")
    T4_koop   = log_data.T4(:);
    Pump_koop = log_data.Pump2(:);

    %% ===== LOAD STREJC =====
    S = load(strejc_file);
    log_data = S.log_data;

    t = datetime(string(log_data.timestamp), 'InputFormat', time_format);
    time_strejc = seconds(t - t(1));
    T4_strejc   = log_data.T4(:);
    Pump_strejc = log_data.Pump2(:);

    %% ===== ALIGN LENGTHS (important for fair metrics + nice plots) =====
    L = min([length(T4_koop), length(T4_strejc), length(Pump_koop), length(Pump_strejc)]);
    T4_koop   = T4_koop(1:L);
    T4_strejc = T4_strejc(1:L);
    Pump_koop = Pump_koop(1:L);
    Pump_strejc = Pump_strejc(1:L);

    % create common x-axis as "time step" (0..L-1)
    step = (0:L-1)';

    %% ===== METRICS =====
    target = x_mean_target;
    bias   = target - x_mean;

    % RMSE
    rmse_koop   = sqrt(mean((T4_koop   - target).^2));
    rmse_strejc = sqrt(mean((T4_strejc - target).^2));
    sum_rmse_koop_all   = sum_rmse_koop_all   + rmse_koop;
    sum_rmse_strejc_all = sum_rmse_strejc_all + rmse_strejc;

    % IAE
    iae_koop   = sum(abs(T4_koop   - target));
    iae_strejc = sum(abs(T4_strejc - target));
    sum_iae_koop_all   = sum_iae_koop_all   + iae_koop;
    sum_iae_strejc_all = sum_iae_strejc_all + iae_strejc;

    % Objective (physical)
    obj_koop = sum(Q_cost*(T4_koop - target).^2 + R_cost*(Pump_koop).^2);
    obj_strejc = sum(Q_cost*(T4_strejc - target).^2 + R_cost*(Pump_strejc).^2);

    % Objective (scaled like MPC but without rescaling every measurement set)
    obj_koop_scaled = sum( ...
        Q_cost*((T4_koop   - bias - x_mean)/x_std).^2 + ...
        R_cost*((Pump_koop - u_mean)/u_std).^2 );

    obj_strejc_scaled = sum( ...
        Q_cost*((T4_strejc - bias - x_mean)/x_std).^2 + ...
        R_cost*((Pump_strejc - u_mean)/u_std).^2 );

    sum_obj_koop_scaled_all   = sum_obj_koop_scaled_all   + obj_koop_scaled;
    sum_obj_strejc_scaled_all = sum_obj_strejc_scaled_all + obj_strejc_scaled;

    % Print per-T0 short summary (optional)
    fprintf('T0=%d: Koop RMSE=%.3f IAE=%.1f ObjS=%.2f | Strejc RMSE=%.3f IAE=%.1f ObjS=%.2f\n', ...
        T0, rmse_koop, iae_koop, obj_koop_scaled, rmse_strejc, iae_strejc, obj_strejc_scaled);

    % Store into global table
    all_rows = [all_rows; ...
        table(T0, "Koopman", rmse_koop, iae_koop, obj_koop, obj_koop_scaled, ...
              'VariableNames', {'T0','Controller','RMSE_T4','IAE','Objective','Objective_Scaled'}); ...
        table(T0, "Strejc", rmse_strejc, iae_strejc, obj_strejc, obj_strejc_scaled, ...
              'VariableNames', {'T0','Controller','RMSE_T4','IAE','Objective','Objective_Scaled'})];

    %% ===== PLOT (like your simulation figure: output + input) =====
    fig = figure('Color','w','Position',[100 100 900 520]);
    tiledlayout(2,1,'TileSpacing','Compact','Padding','Compact');

    % --- OUTPUT ---
    nexttile;
    plot(step, T4_koop, 'm', 'LineWidth', 2); hold on;
    plot(step, T4_strejc, 'b--', 'LineWidth', 2);
    yline(target, 'k-', 'LineWidth', 1.2);
    grid on; grid minor;
    ylabel('Outlet temperature ($^\circ$C)');
    title(sprintf('Closed-loop response (start %d$^\\circ$C)', T0));
    legend('Koopman MPC','Strejc MPC','Steady-state','Location','best');
    ylim([min([T4_koop;T4_strejc;target])-1, max([T4_koop;T4_strejc;target])+1]);

    % --- INPUT ---
    nexttile;
    stairs(step, Pump_koop, 'm', 'LineWidth', 2); hold on;
    stairs(step, Pump_strejc, 'b--', 'LineWidth', 2);
    grid on; grid minor;
    xlabel('Time step');
    ylabel('Pump speed (\%)');
    title('Control input');
    legend('Koopman MPC','Strejc MPC','Location','best');

    % small text with metrics on the plot (optional)
%     annotation(fig,'textbox',[0.12 0.01 0.86 0.06], ...
%         'String', sprintf('Koopman: RMSE=%.3f, IAE=%.1f, ObjS=%.2f   |   Strejc: RMSE=%.3f, IAE=%.1f, ObjS=%.2f', ...
%             rmse_koop, iae_koop, obj_koop_scaled, rmse_strejc, iae_strejc, obj_strejc_scaled), ...
%         'EdgeColor','none','Interpreter','latex');

    % save
    out_png = sprintf('figs/compare_cl_T0_%d.png', T0);
    saveas(fig, out_png);

end

%% ===== FINAL GLOBAL RESULT =====
metrics_total = table( ...
    {'Koopman'; 'Strejc'}, ...
    [sum_rmse_koop_all; sum_rmse_strejc_all], ...
    [sum_iae_koop_all;  sum_iae_strejc_all], ...
    [sum_obj_koop_scaled_all; sum_obj_strejc_scaled_all], ...
    'VariableNames', {'Controller','Total_RMSE','Total_IAE','Total_Objective_Scaled'} );

disp('===== FINAL SUM OVER ALL INITIAL CONDITIONS =====');
disp(metrics_total);

%% ===== OPTIONAL: show the per-T0 table nicely =====
disp('===== PER-T0 METRICS (Koopman + Strejc) =====');
disp(all_rows);

% If you want one row per T0 (two controllers in columns), we can pivot later.
