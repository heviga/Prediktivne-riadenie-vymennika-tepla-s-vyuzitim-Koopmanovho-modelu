%% Auto-plot + metrics for ALL start temperatures
% Fixed transient evaluation over first 100 samples only
clc; clear; close all;

project_root = 'C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\LS2026';
addpath(genpath(project_root));

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');

%% ===== SCRIPT PATH / OUTPUT PATHS =====
script_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(script_dir, 'figs', 'transient_first100');
res_dir = fullfile(script_dir, 'results', 'transient_first100');

if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

if ~exist(res_dir, 'dir')
    mkdir(res_dir);
end

%% ===== SCALING CONSTANTS =====
x_mean = 59.0676;
x_std  = 6.9122;

u_mean = 65.8447; %#ok<NASGU>
u_std  = 22.9062;

%% ===== SETTINGS =====
time_format = 'yyyy-MM-dd HH:mm:ss.SSS';

Q_cost = 10;
R_cost = 1;

temps = [45, 50, 55, 58, 60, 62, 66, 68];
x_mean_target = 59.0676 + abs(59.0676 - 63.8084);

Ntr_fixed = 100;

%% ===== GLOBAL ACCUMULATORS =====
sum_obj_koop_scaled_all   = 0;
sum_obj_strejc_scaled_all = 0;

sum_rmse_koop_all   = 0;
sum_rmse_strejc_all = 0;

sum_iae_koop_all   = 0;
sum_iae_strejc_all = 0;

sum_iae_fix_koop_all    = 0;
sum_iae_fix_strejc_all  = 0;
sum_ise_fix_koop_all    = 0;
sum_ise_fix_strejc_all  = 0;
sum_itae_fix_koop_all   = 0;
sum_itae_fix_strejc_all = 0;
sum_rmse_fix_koop_all   = 0;
sum_rmse_fix_strejc_all = 0;
sum_peak_fix_koop_all   = 0;
sum_peak_fix_strejc_all = 0;
sum_over_fix_koop_all   = 0;
sum_over_fix_strejc_all = 0;
sum_under_fix_koop_all  = 0;
sum_under_fix_strejc_all = 0;
sum_tvu_fix_koop_all    = 0;
sum_tvu_fix_strejc_all  = 0;
sum_dumax_fix_koop_all  = 0;
sum_dumax_fix_strejc_all = 0;
sum_uen_fix_koop_all    = 0;
sum_uen_fix_strejc_all  = 0;

%% ===== VECTORS FOR METRIC PLOTS =====
T0_vec = temps(:);

RMSE_K = nan(length(temps),1); RMSE_S = nan(length(temps),1);
IAE_K  = nan(length(temps),1); IAE_S  = nan(length(temps),1);
OBJ_K  = nan(length(temps),1); OBJ_S  = nan(length(temps),1);

IAE100_K = nan(length(temps),1);  IAE100_S = nan(length(temps),1);
ISE100_K = nan(length(temps),1);  ISE100_S = nan(length(temps),1);
ITAE100_K = nan(length(temps),1); ITAE100_S = nan(length(temps),1);
RMSE100_K = nan(length(temps),1); RMSE100_S = nan(length(temps),1);
PEAK100_K = nan(length(temps),1); PEAK100_S = nan(length(temps),1);
OVER100_K = nan(length(temps),1); OVER100_S = nan(length(temps),1);
UNDER100_K = nan(length(temps),1); UNDER100_S = nan(length(temps),1);
TVU100_K = nan(length(temps),1); TVU100_S = nan(length(temps),1);
DUMAX100_K = nan(length(temps),1); DUMAX100_S = nan(length(temps),1);
UEN100_K = nan(length(temps),1); UEN100_S = nan(length(temps),1);

%% ===== TABLE STORAGE =====
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

    t = datetime(string(log_data.timestamp), 'InputFormat', time_format); %#ok<NASGU>
    T4_koop   = log_data.T4(:);
    Pump_koop = log_data.Pump2(:);

    %% ===== LOAD STREJC =====
    S = load(strejc_file);
    log_data = S.log_data;

    t = datetime(string(log_data.timestamp), 'InputFormat', time_format); %#ok<NASGU>
    T4_strejc   = log_data.T4(:);
    Pump_strejc = log_data.Pump2(:);

    %% ===== ALIGN LENGTHS =====
    L = min([length(T4_koop), length(T4_strejc), length(Pump_koop), length(Pump_strejc)]);

    T4_koop     = T4_koop(1:L);
    T4_strejc   = T4_strejc(1:L);
    Pump_koop   = Pump_koop(1:L);
    Pump_strejc = Pump_strejc(1:L);

    step = (0:L-1)';

    %% ===== TARGET =====
    target = x_mean_target;
    bias   = target - x_mean;

    %% ===== FULL-RESPONSE METRICS =====
    rmse_koop   = sqrt(mean((T4_koop   - target).^2));
    rmse_strejc = sqrt(mean((T4_strejc - target).^2));

    iae_koop   = sum(abs(T4_koop   - target));
    iae_strejc = sum(abs(T4_strejc - target));

    obj_koop   = sum(Q_cost*(T4_koop   - target).^2 + R_cost*(Pump_koop).^2);
    obj_strejc = sum(Q_cost*(T4_strejc - target).^2 + R_cost*(Pump_strejc).^2);

    obj_koop_scaled = sum( ...
        Q_cost*((T4_koop   - bias - x_mean)/x_std).^2 + ...
        R_cost*((Pump_koop - 65.8447)/u_std).^2 );

    obj_strejc_scaled = sum( ...
        Q_cost*((T4_strejc - bias - x_mean)/x_std).^2 + ...
        R_cost*((Pump_strejc - 65.8447)/u_std).^2 );

    sum_rmse_koop_all = sum_rmse_koop_all + rmse_koop;
    sum_rmse_strejc_all = sum_rmse_strejc_all + rmse_strejc;
    sum_iae_koop_all = sum_iae_koop_all + iae_koop;
    sum_iae_strejc_all = sum_iae_strejc_all + iae_strejc;
    sum_obj_koop_scaled_all = sum_obj_koop_scaled_all + obj_koop_scaled;
    sum_obj_strejc_scaled_all = sum_obj_strejc_scaled_all + obj_strejc_scaled;

    %% ===== FIRST 100 SAMPLES METRICS =====
    fix_koop   = compute_firstN_metrics(T4_koop, Pump_koop, target, Ntr_fixed);
    fix_strejc = compute_firstN_metrics(T4_strejc, Pump_strejc, target, Ntr_fixed);

    sum_iae_fix_koop_all    = sum_iae_fix_koop_all    + fix_koop.IAE;
    sum_iae_fix_strejc_all  = sum_iae_fix_strejc_all  + fix_strejc.IAE;
    sum_ise_fix_koop_all    = sum_ise_fix_koop_all    + fix_koop.ISE;
    sum_ise_fix_strejc_all  = sum_ise_fix_strejc_all  + fix_strejc.ISE;
    sum_itae_fix_koop_all   = sum_itae_fix_koop_all   + fix_koop.ITAE;
    sum_itae_fix_strejc_all = sum_itae_fix_strejc_all + fix_strejc.ITAE;
    sum_rmse_fix_koop_all   = sum_rmse_fix_koop_all   + fix_koop.RMSE;
    sum_rmse_fix_strejc_all = sum_rmse_fix_strejc_all + fix_strejc.RMSE;
    sum_peak_fix_koop_all   = sum_peak_fix_koop_all   + fix_koop.PeakDeviation;
    sum_peak_fix_strejc_all = sum_peak_fix_strejc_all + fix_strejc.PeakDeviation;
    sum_over_fix_koop_all   = sum_over_fix_koop_all   + fix_koop.Overshoot;
    sum_over_fix_strejc_all = sum_over_fix_strejc_all + fix_strejc.Overshoot;
    sum_under_fix_koop_all  = sum_under_fix_koop_all  + fix_koop.Undershoot;
    sum_under_fix_strejc_all = sum_under_fix_strejc_all + fix_strejc.Undershoot;
    sum_tvu_fix_koop_all    = sum_tvu_fix_koop_all    + fix_koop.TVu;
    sum_tvu_fix_strejc_all  = sum_tvu_fix_strejc_all  + fix_strejc.TVu;
    sum_dumax_fix_koop_all  = sum_dumax_fix_koop_all  + fix_koop.DeltaUMax;
    sum_dumax_fix_strejc_all = sum_dumax_fix_strejc_all + fix_strejc.DeltaUMax;
    sum_uen_fix_koop_all    = sum_uen_fix_koop_all    + fix_koop.ControlEnergy;
    sum_uen_fix_strejc_all  = sum_uen_fix_strejc_all  + fix_strejc.ControlEnergy;

    %% ===== PRINT =====
    fprintf('T0=%d | FULL: Koop RMSE=%.3f IAE=%.1f ObjS=%.2f | Strejc RMSE=%.3f IAE=%.1f ObjS=%.2f\n', ...
        T0, rmse_koop, iae_koop, obj_koop_scaled, rmse_strejc, iae_strejc, obj_strejc_scaled);

    fprintf(['T0=%d | FIRST %d SAMPLES: Koop RMSE=%.3f IAE=%.2f ITAE=%.2f Peak=%.2f TVu=%.2f | ' ...
             'Strejc RMSE=%.3f IAE=%.2f ITAE=%.2f Peak=%.2f TVu=%.2f\n'], ...
        T0, Ntr_fixed, ...
        fix_koop.RMSE, fix_koop.IAE, fix_koop.ITAE, fix_koop.PeakDeviation, fix_koop.TVu, ...
        fix_strejc.RMSE, fix_strejc.IAE, fix_strejc.ITAE, fix_strejc.PeakDeviation, fix_strejc.TVu);

%% ===== STORE FOR SUMMARY PLOTS =====
RMSE_K(k) = rmse_koop;       RMSE_S(k) = rmse_strejc;
IAE_K(k)  = iae_koop;        IAE_S(k)  = iae_strejc;
OBJ_K(k)  = obj_koop_scaled; OBJ_S(k)  = obj_strejc_scaled;

IAE100_K(k) = fix_koop.IAE;   IAE100_S(k) = fix_strejc.IAE;
ISE100_K(k) = fix_koop.ISE;   ISE100_S(k) = fix_strejc.ISE;
ITAE100_K(k) = fix_koop.ITAE; ITAE100_S(k) = fix_strejc.ITAE;
RMSE100_K(k) = fix_koop.RMSE; RMSE100_S(k) = fix_strejc.RMSE;
PEAK100_K(k) = fix_koop.PeakDeviation; PEAK100_S(k) = fix_strejc.PeakDeviation;
OVER100_K(k) = fix_koop.Overshoot;     OVER100_S(k) = fix_strejc.Overshoot;
UNDER100_K(k) = fix_koop.Undershoot;   UNDER100_S(k) = fix_strejc.Undershoot;
TVU100_K(k) = fix_koop.TVu;            TVU100_S(k) = fix_strejc.TVu;
DUMAX100_K(k) = fix_koop.DeltaUMax;    DUMAX100_S(k) = fix_strejc.DeltaUMax;
UEN100_K(k) = fix_koop.ControlEnergy;  UEN100_S(k) = fix_strejc.ControlEnergy;

    %% ===== STORE =====
    all_rows = [all_rows; ...
        table(T0, "Koopman", ...
              rmse_koop, iae_koop, obj_koop, obj_koop_scaled, ...
              fix_koop.N, fix_koop.IAE, fix_koop.ISE, fix_koop.ITAE, fix_koop.RMSE, ...
              fix_koop.PeakDeviation, fix_koop.Overshoot, fix_koop.Undershoot, ...
              fix_koop.TVu, fix_koop.DeltaUMax, fix_koop.ControlEnergy, ...
              'VariableNames', {'T0','Controller','RMSE_T4','IAE','Objective','Objective_Scaled', ...
              'TransientLength','IAE_FirstN','ISE_FirstN','ITAE_FirstN','RMSE_FirstN', ...
              'PeakDeviation_FirstN','Overshoot_FirstN','Undershoot_FirstN', ...
              'TVu_FirstN','DeltaUMax_FirstN','ControlEnergy_FirstN'}); ...
        table(T0, "Strejc", ...
              rmse_strejc, iae_strejc, obj_strejc, obj_strejc_scaled, ...
              fix_strejc.N, fix_strejc.IAE, fix_strejc.ISE, fix_strejc.ITAE, fix_strejc.RMSE, ...
              fix_strejc.PeakDeviation, fix_strejc.Overshoot, fix_strejc.Undershoot, ...
              fix_strejc.TVu, fix_strejc.DeltaUMax, fix_strejc.ControlEnergy, ...
              'VariableNames', {'T0','Controller','RMSE_T4','IAE','Objective','Objective_Scaled', ...
              'TransientLength','IAE_FirstN','ISE_FirstN','ITAE_FirstN','RMSE_FirstN', ...
              'PeakDeviation_FirstN','Overshoot_FirstN','Undershoot_FirstN', ...
              'TVu_FirstN','DeltaUMax_FirstN','ControlEnergy_FirstN'})];

   %% ===== PLOT =====
fig = figure('Color','w','Position',[100 100 900 520]);
tiledlayout(2,1,'TileSpacing','Compact','Padding','Compact');

idx_plot = 1:fix_koop.N;
step_plot = step(idx_plot);

% OUTPUT
nexttile;
h1 = plot(step_plot, T4_koop(idx_plot), 'm', 'LineWidth', 2); hold on;
h2 = plot(step_plot, T4_strejc(idx_plot), 'b--', 'LineWidth', 2);
h3 = yline(target, 'k-', 'LineWidth', 1.2);

xline(fix_koop.N-1, 'k:', 'LineWidth', 1.2, 'HandleVisibility','off');

grid on; grid minor;
ylabel('Outlet temperature ($^\circ$C)');
title(sprintf('First %d samples only (start %d$^\\circ$C)', fix_koop.N, T0));
legend([h1 h2 h3], {'Koopman MPC','Strejc MPC','Target'}, 'Location','best');

ymin_plot = min([T4_koop(idx_plot); T4_strejc(idx_plot); target]) - 1;
ymax_plot = max([T4_koop(idx_plot); T4_strejc(idx_plot); target]) + 1;
ylim([ymin_plot, ymax_plot]);
xlim([0, step_plot(end)]);

% INPUT
nexttile;
h4 = stairs(step_plot, Pump_koop(idx_plot), 'm', 'LineWidth', 2); hold on;
h5 = stairs(step_plot, Pump_strejc(idx_plot), 'b--', 'LineWidth', 2);

xline(fix_koop.N-1, 'k:', 'LineWidth', 1.2, 'HandleVisibility','off');

grid on; grid minor;
xlabel('Time step');
ylabel('Pump speed (\%)');
title('Control input during first 100 samples');
legend([h4 h5], {'Koopman MPC','Strejc MPC'}, 'Location','best');
xlim([0, step_plot(end)]);

out_png = fullfile(fig_dir, sprintf('compare_cl_T0_%d.png', T0));
saveas(fig, out_png);
end

%% ===== TOTAL TABLE =====
metrics_total = table( ...
    {'Koopman'; 'Strejc'}, ...
    [sum_rmse_koop_all; sum_rmse_strejc_all], ...
    [sum_iae_koop_all; sum_iae_strejc_all], ...
    [sum_obj_koop_scaled_all; sum_obj_strejc_scaled_all], ...
    [sum_iae_fix_koop_all; sum_iae_fix_strejc_all], ...
    [sum_ise_fix_koop_all; sum_ise_fix_strejc_all], ...
    [sum_itae_fix_koop_all; sum_itae_fix_strejc_all], ...
    [sum_rmse_fix_koop_all; sum_rmse_fix_strejc_all], ...
    [sum_peak_fix_koop_all; sum_peak_fix_strejc_all], ...
    [sum_over_fix_koop_all; sum_over_fix_strejc_all], ...
    [sum_under_fix_koop_all; sum_under_fix_strejc_all], ...
    [sum_tvu_fix_koop_all; sum_tvu_fix_strejc_all], ...
    [sum_dumax_fix_koop_all; sum_dumax_fix_strejc_all], ...
    [sum_uen_fix_koop_all; sum_uen_fix_strejc_all], ...
    'VariableNames', {'Controller', ...
    'Total_RMSE','Total_IAE','Total_Objective_Scaled', ...
    'Total_IAE_FirstN','Total_ISE_FirstN','Total_ITAE_FirstN','Total_RMSE_FirstN', ...
    'Total_PeakDeviation_FirstN','Total_Overshoot_FirstN','Total_Undershoot_FirstN', ...
    'Total_TVu_FirstN','Total_DeltaUMax_FirstN','Total_ControlEnergy_FirstN'} );

disp('===== FINAL SUM OVER ALL INITIAL CONDITIONS =====');
disp(metrics_total);

disp('===== PER-T0 METRICS =====');
disp(all_rows);

%% ===== AVERAGE TABLE =====
n_cases = length(temps);
metrics_avg = metrics_total;
metrics_avg{:, 2:end} = metrics_avg{:, 2:end} / n_cases;

disp('===== AVERAGE PER INITIAL CONDITION =====');
disp(metrics_avg);

%% ===== SAVE =====
writetable(all_rows, fullfile(res_dir, 'per_T0_metrics_first100.csv'));
writetable(metrics_total, fullfile(res_dir, 'metrics_total_first100.csv'));
writetable(metrics_avg, fullfile(res_dir, 'metrics_avg_first100.csv'));

disp('Saved:');
disp(fullfile(res_dir, 'per_T0_metrics_first100.csv'));
disp(fullfile(res_dir, 'metrics_total_first100.csv'));
disp(fullfile(res_dir, 'metrics_avg_first100.csv'));


%% ===== METRICS vs INITIAL CONDITION =====
plot_metric_figure(T0_vec, RMSE_K, RMSE_S, 'RMSE', 'RMSE ($^\circ$C)', ...
    fullfile(fig_dir, 'metric_RMSE.png'));

plot_metric_figure(T0_vec, IAE_K, IAE_S, 'IAE', 'IAE', ...
    fullfile(fig_dir, 'metric_IAE.png'));

plot_metric_figure(T0_vec, OBJ_K, OBJ_S, 'Objective', 'Objective scaled', ...
    fullfile(fig_dir, 'metric_Objective.png'));

plot_metric_figure(T0_vec, IAE100_K, IAE100_S, 'IAE first 100 samples', 'IAE', ...
    fullfile(fig_dir, 'metric_IAE_first100.png'));

plot_metric_figure(T0_vec, ISE100_K, ISE100_S, 'ISE first 100 samples', 'ISE', ...
    fullfile(fig_dir, 'metric_ISE_first100.png'));

plot_metric_figure(T0_vec, ITAE100_K, ITAE100_S, 'ITAE first 100 samples', 'ITAE', ...
    fullfile(fig_dir, 'metric_ITAE_first100.png'));

plot_metric_figure(T0_vec, RMSE100_K, RMSE100_S, 'RMSE first 100 samples', 'RMSE ($^\circ$C)', ...
    fullfile(fig_dir, 'metric_RMSE_first100.png'));

plot_metric_figure(T0_vec, PEAK100_K, PEAK100_S, 'Peak deviation first 100 samples', 'Peak deviation ($^\circ$C)', ...
    fullfile(fig_dir, 'metric_Peak_first100.png'));

plot_metric_figure(T0_vec, OVER100_K, OVER100_S, 'Overshoot first 100 samples', 'Overshoot ($^\circ$C)', ...
    fullfile(fig_dir, 'metric_Overshoot_first100.png'));

plot_metric_figure(T0_vec, UNDER100_K, UNDER100_S, 'Undershoot first 100 samples', 'Undershoot ($^\circ$C)', ...
    fullfile(fig_dir, 'metric_Undershoot_first100.png'));

plot_metric_figure(T0_vec, TVU100_K, TVU100_S, 'TVu first 100 samples', 'TVu', ...
    fullfile(fig_dir, 'metric_TVu_first100.png'));

plot_metric_figure(T0_vec, DUMAX100_K, DUMAX100_S, 'Maximum input increment first 100 samples', 'Max $|\Delta u|$', ...
    fullfile(fig_dir, 'metric_DeltaUMax_first100.png'));

plot_metric_figure(T0_vec, UEN100_K, UEN100_S, 'Control energy first 100 samples', 'Control energy', ...
    fullfile(fig_dir, 'metric_ControlEnergy_first100.png'));
%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================

function metrics = compute_firstN_metrics(y, u, target, Ntr)
    y = y(:);
    u = u(:);

    N = min([length(y), length(u), Ntr]);

    y_tr = y(1:N);
    u_tr = u(1:N);
    e = y_tr - target;

    metrics.N    = N;
    metrics.IAE  = sum(abs(e));
    metrics.ISE  = sum(e.^2);
    metrics.ITAE = sum((0:N-1)' .* abs(e));
    metrics.RMSE = sqrt(mean(e.^2));

    [~, idx_peak] = max(abs(e));
    metrics.PeakDeviation = abs(e(idx_peak));
    metrics.PeakIndex     = idx_peak;

    metrics.Overshoot  = max(0, max(y_tr - target));
    metrics.Undershoot = max(0, max(target - y_tr));

    if length(u_tr) > 1
        du = diff(u_tr);
        metrics.TVu       = sum(abs(du));
        metrics.DeltaUMax = max(abs(du));
    else
        metrics.TVu       = 0;
        metrics.DeltaUMax = 0;
    end

    metrics.ControlEnergy = sum(u_tr.^2);
end


function plot_metric_figure(T0_vec, Yk, Ys, ttl, ylab, save_path)
    figM = figure('Color','w','Position',[100 100 520 360]);
    plot(T0_vec, Yk, 'm-o', 'LineWidth', 2); hold on;
    plot(T0_vec, Ys, 'b-s', 'LineWidth', 2);
    grid on; grid minor;
    xlabel('Initial temperature $T_0$ ($^\circ$C)');
    ylabel(ylab);
    title(ttl);
    legend('Koopman','Strejc','Location','best');
    saveas(figM, save_path);
end