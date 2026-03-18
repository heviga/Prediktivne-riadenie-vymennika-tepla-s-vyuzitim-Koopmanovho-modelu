%% Auto-plot + metrics for ALL start temperatures
% Dynamic transient evaluation based on settling detection only
clc; clear; close all;

project_root = 'C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\LS2026';
addpath(genpath(project_root));

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');

%% ===== SCRIPT PATH / OUTPUT PATHS =====
script_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(script_dir, 'figs', 'transient_dynamic');
res_dir = fullfile(script_dir, 'results', 'transient_dynamic');

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

% target used in plots / evaluation
x_mean_target = 59.0676 + abs(59.0676 - 63.8084);

%% ===== DYNAMIC TRANSIENT SETTINGS =====
settling_band = 0.4;       % +/- band around target
min_hold = 10;             % window length
inside_fraction_required = 0.8;   % at least 80 percent of samples in window inside band

%% ===== GLOBAL ACCUMULATORS =====
sum_obj_koop_scaled_all   = 0;
sum_obj_strejc_scaled_all = 0;

sum_rmse_koop_all   = 0;
sum_rmse_strejc_all = 0;

sum_iae_koop_all   = 0;
sum_iae_strejc_all = 0;

sum_ts_koop_all        = 0;
sum_ts_strejc_all      = 0;
sum_iae_tr_koop_all    = 0;
sum_iae_tr_strejc_all  = 0;
sum_ise_tr_koop_all    = 0;
sum_ise_tr_strejc_all  = 0;
sum_itae_tr_koop_all   = 0;
sum_itae_tr_strejc_all = 0;
sum_rmse_tr_koop_all   = 0;
sum_rmse_tr_strejc_all = 0;
sum_peak_koop_all      = 0;
sum_peak_strejc_all    = 0;
sum_over_koop_all      = 0;
sum_over_strejc_all    = 0;
sum_under_koop_all     = 0;
sum_under_strejc_all   = 0;
sum_tvu_koop_all       = 0;
sum_tvu_strejc_all     = 0;
sum_dumax_koop_all     = 0;
sum_dumax_strejc_all   = 0;
sum_uenergy_koop_all   = 0;
sum_uenergy_strejc_all = 0;

count_settled_koop   = 0;
count_settled_strejc = 0;

%% ===== TABLE STORAGE =====
all_rows = [];

%% ===== VECTORS FOR METRIC PLOTS =====
T0_vec = temps(:);

RMSE_K = nan(length(temps),1); RMSE_S = nan(length(temps),1);
IAE_K  = nan(length(temps),1); IAE_S  = nan(length(temps),1);
OBJ_K  = nan(length(temps),1); OBJ_S  = nan(length(temps),1);

TS_K   = nan(length(temps),1); TS_S   = nan(length(temps),1);
IAEtr_K = nan(length(temps),1); IAEtr_S = nan(length(temps),1);
ISEtr_K = nan(length(temps),1); ISEtr_S = nan(length(temps),1);
ITAEtr_K = nan(length(temps),1); ITAEtr_S = nan(length(temps),1);
RMSEtr_K = nan(length(temps),1); RMSEtr_S = nan(length(temps),1);
PEAK_K = nan(length(temps),1); PEAK_S = nan(length(temps),1);
OVER_K = nan(length(temps),1); OVER_S = nan(length(temps),1);
UNDER_K = nan(length(temps),1); UNDER_S = nan(length(temps),1);
TVU_K = nan(length(temps),1); TVU_S = nan(length(temps),1);
DUMAX_K = nan(length(temps),1); DUMAX_S = nan(length(temps),1);
UEN_K = nan(length(temps),1); UEN_S = nan(length(temps),1);
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

    %% ===== DYNAMIC TRANSIENT METRICS =====
    [ts_idx_koop, tr_koop] = compute_dynamic_transient_metrics( ...
        T4_koop, Pump_koop, target, settling_band, min_hold, inside_fraction_required);

    [ts_idx_strejc, tr_strejc] = compute_dynamic_transient_metrics( ...
        T4_strejc, Pump_strejc, target, settling_band, min_hold, inside_fraction_required);

    if ~isnan(tr_koop.SettlingTime)
        sum_ts_koop_all = sum_ts_koop_all + tr_koop.SettlingTime;
        count_settled_koop = count_settled_koop + 1;
    end
    if ~isnan(tr_strejc.SettlingTime)
        sum_ts_strejc_all = sum_ts_strejc_all + tr_strejc.SettlingTime;
        count_settled_strejc = count_settled_strejc + 1;
    end

    sum_iae_tr_koop_all    = sum_iae_tr_koop_all    + tr_koop.IAE_transient;
    sum_iae_tr_strejc_all  = sum_iae_tr_strejc_all  + tr_strejc.IAE_transient;
    sum_ise_tr_koop_all    = sum_ise_tr_koop_all    + tr_koop.ISE_transient;
    sum_ise_tr_strejc_all  = sum_ise_tr_strejc_all  + tr_strejc.ISE_transient;
    sum_itae_tr_koop_all   = sum_itae_tr_koop_all   + tr_koop.ITAE_transient;
    sum_itae_tr_strejc_all = sum_itae_tr_strejc_all + tr_strejc.ITAE_transient;
    sum_rmse_tr_koop_all   = sum_rmse_tr_koop_all   + tr_koop.RMSE_transient;
    sum_rmse_tr_strejc_all = sum_rmse_tr_strejc_all + tr_strejc.RMSE_transient;
    sum_peak_koop_all      = sum_peak_koop_all      + tr_koop.PeakDeviation;
    sum_peak_strejc_all    = sum_peak_strejc_all    + tr_strejc.PeakDeviation;
    sum_over_koop_all      = sum_over_koop_all      + tr_koop.Overshoot;
    sum_over_strejc_all    = sum_over_strejc_all    + tr_strejc.Overshoot;
    sum_under_koop_all     = sum_under_koop_all     + tr_koop.Undershoot;
    sum_under_strejc_all   = sum_under_strejc_all   + tr_strejc.Undershoot;
    sum_tvu_koop_all       = sum_tvu_koop_all       + tr_koop.TVu_transient;
    sum_tvu_strejc_all     = sum_tvu_strejc_all     + tr_strejc.TVu_transient;
    sum_dumax_koop_all     = sum_dumax_koop_all     + tr_koop.DeltaUMax;
    sum_dumax_strejc_all   = sum_dumax_strejc_all   + tr_strejc.DeltaUMax;
    sum_uenergy_koop_all   = sum_uenergy_koop_all   + tr_koop.ControlEnergy;
    sum_uenergy_strejc_all = sum_uenergy_strejc_all + tr_strejc.ControlEnergy;


    %% ===== STORE FOR SUMMARY PLOTS =====
RMSE_K(k) = rmse_koop;         RMSE_S(k) = rmse_strejc;
IAE_K(k)  = iae_koop;          IAE_S(k)  = iae_strejc;
OBJ_K(k)  = obj_koop_scaled;   OBJ_S(k)  = obj_strejc_scaled;

TS_K(k) = tr_koop.SettlingTime;     TS_S(k) = tr_strejc.SettlingTime;
IAEtr_K(k) = tr_koop.IAE_transient; IAEtr_S(k) = tr_strejc.IAE_transient;
ISEtr_K(k) = tr_koop.ISE_transient; ISEtr_S(k) = tr_strejc.ISE_transient;
ITAEtr_K(k) = tr_koop.ITAE_transient; ITAEtr_S(k) = tr_strejc.ITAE_transient;
RMSEtr_K(k) = tr_koop.RMSE_transient; RMSEtr_S(k) = tr_strejc.RMSE_transient;
PEAK_K(k) = tr_koop.PeakDeviation;  PEAK_S(k) = tr_strejc.PeakDeviation;
OVER_K(k) = tr_koop.Overshoot;      OVER_S(k) = tr_strejc.Overshoot;
UNDER_K(k) = tr_koop.Undershoot;    UNDER_S(k) = tr_strejc.Undershoot;
TVU_K(k) = tr_koop.TVu_transient;   TVU_S(k) = tr_strejc.TVu_transient;
DUMAX_K(k) = tr_koop.DeltaUMax;     DUMAX_S(k) = tr_strejc.DeltaUMax;
UEN_K(k) = tr_koop.ControlEnergy;   UEN_S(k) = tr_strejc.ControlEnergy;
    %% ===== PRINT =====
    fprintf('T0=%d | FULL: Koop RMSE=%.3f IAE=%.1f ObjS=%.2f | Strejc RMSE=%.3f IAE=%.1f ObjS=%.2f\n', ...
        T0, rmse_koop, iae_koop, obj_koop_scaled, rmse_strejc, iae_strejc, obj_strejc_scaled);

    fprintf(['T0=%d | DYNAMIC TRANSIENT: Koop Ts=%s IAEtr=%.2f ITAEtr=%.2f Peak=%.2f TVu=%.2f | ' ...
             'Strejc Ts=%s IAEtr=%.2f ITAEtr=%.2f Peak=%.2f TVu=%.2f\n'], ...
        T0, ...
        num2str_or_nan(tr_koop.SettlingTime), tr_koop.IAE_transient, tr_koop.ITAE_transient, tr_koop.PeakDeviation, tr_koop.TVu_transient, ...
        num2str_or_nan(tr_strejc.SettlingTime), tr_strejc.IAE_transient, tr_strejc.ITAE_transient, tr_strejc.PeakDeviation, tr_strejc.TVu_transient);

    %% ===== STORE =====
    all_rows = [all_rows; ...
        table(T0, "Koopman", ...
              rmse_koop, iae_koop, obj_koop, obj_koop_scaled, ...
              tr_koop.SettlingTime, tr_koop.SettlingIndex, tr_koop.SettlingFound, tr_koop.TransientLength, ...
              tr_koop.IAE_transient, tr_koop.ISE_transient, tr_koop.ITAE_transient, tr_koop.RMSE_transient, ...
              tr_koop.PeakDeviation, tr_koop.Overshoot, tr_koop.Undershoot, ...
              tr_koop.TVu_transient, tr_koop.DeltaUMax, tr_koop.ControlEnergy, ...
              'VariableNames', {'T0','Controller','RMSE_T4','IAE','Objective','Objective_Scaled', ...
              'SettlingTime','SettlingIndex','SettlingFound','TransientLength', ...
              'IAE_Transient','ISE_Transient','ITAE_Transient','RMSE_Transient', ...
              'PeakDeviation','Overshoot','Undershoot', ...
              'TVu_Transient','DeltaUMax_Transient','ControlEnergy_Transient'}); ...
        table(T0, "Strejc", ...
              rmse_strejc, iae_strejc, obj_strejc, obj_strejc_scaled, ...
              tr_strejc.SettlingTime, tr_strejc.SettlingIndex, tr_strejc.SettlingFound, tr_strejc.TransientLength, ...
              tr_strejc.IAE_transient, tr_strejc.ISE_transient, tr_strejc.ITAE_transient, tr_strejc.RMSE_transient, ...
              tr_strejc.PeakDeviation, tr_strejc.Overshoot, tr_strejc.Undershoot, ...
              tr_strejc.TVu_transient, tr_strejc.DeltaUMax, tr_strejc.ControlEnergy, ...
              'VariableNames', {'T0','Controller','RMSE_T4','IAE','Objective','Objective_Scaled', ...
              'SettlingTime','SettlingIndex','SettlingFound','TransientLength', ...
              'IAE_Transient','ISE_Transient','ITAE_Transient','RMSE_Transient', ...
              'PeakDeviation','Overshoot','Undershoot', ...
              'TVu_Transient','DeltaUMax_Transient','ControlEnergy_Transient'})];

 %% ===== PLOT =====
fig = figure('Color','w','Position',[100 100 900 520]);
tiledlayout(2,1,'TileSpacing','Compact','Padding','Compact');

% indices for plotting only evaluated transient
idx_plot_koop   = 1:tr_koop.TransientLength;
idx_plot_strejc = 1:tr_strejc.TransientLength;

step_koop   = step(idx_plot_koop);
step_strejc = step(idx_plot_strejc);

% OUTPUT
nexttile;
h1 = plot(step_koop, T4_koop(idx_plot_koop), 'm', 'LineWidth', 2); hold on;
h2 = plot(step_strejc, T4_strejc(idx_plot_strejc), 'b--', 'LineWidth', 2);
h3 = yline(target, 'k-', 'LineWidth', 1.2);

yline(target + settling_band, 'k:', 'LineWidth', 0.8, 'HandleVisibility','off');
yline(target - settling_band, 'k:', 'LineWidth', 0.8, 'HandleVisibility','off');

if ~isnan(ts_idx_koop)
    xline(ts_idx_koop-1, 'm:', 'LineWidth', 1.2, 'HandleVisibility','off');
end
if ~isnan(ts_idx_strejc)
    xline(ts_idx_strejc-1, 'b:', 'LineWidth', 1.2, 'HandleVisibility','off');
end

grid on; grid minor;
ylabel('Outlet temperature ($^\circ$C)');
title(sprintf('Dynamic transient only (start %d$^\\circ$C)', T0));
legend([h1 h2 h3], {'Koopman MPC','Strejc MPC','Target'}, 'Location','best');

ymin_plot = min([T4_koop(idx_plot_koop); T4_strejc(idx_plot_strejc); target]) - 1;
ymax_plot = max([T4_koop(idx_plot_koop); T4_strejc(idx_plot_strejc); target]) + 1;
ylim([ymin_plot, ymax_plot]);

xmax_plot = max([step_koop; step_strejc]);
xlim([0, xmax_plot]);

% INPUT
nexttile;
h4 = stairs(step_koop, Pump_koop(idx_plot_koop), 'm', 'LineWidth', 2); hold on;
h5 = stairs(step_strejc, Pump_strejc(idx_plot_strejc), 'b--', 'LineWidth', 2);

if ~isnan(ts_idx_koop)
    xline(ts_idx_koop-1, 'm:', 'LineWidth', 1.2, 'HandleVisibility','off');
end
if ~isnan(ts_idx_strejc)
    xline(ts_idx_strejc-1, 'b:', 'LineWidth', 1.2, 'HandleVisibility','off');
end

grid on; grid minor;
xlabel('Time step');
ylabel('Pump speed (\%)');
title('Control input during evaluated transient');
legend([h4 h5], {'Koopman MPC','Strejc MPC'}, 'Location','best');
xlim([0, xmax_plot]);

out_png = fullfile(fig_dir, sprintf('compare_cl_T0_%d.png', T0));
saveas(fig, out_png);
end

%% ===== TOTAL TABLE =====
metrics_total = table( ...
    {'Koopman'; 'Strejc'}, ...
    [sum_rmse_koop_all; sum_rmse_strejc_all], ...
    [sum_iae_koop_all; sum_iae_strejc_all], ...
    [sum_obj_koop_scaled_all; sum_obj_strejc_scaled_all], ...
    [count_settled_koop; count_settled_strejc], ...
    [sum_ts_koop_all; sum_ts_strejc_all], ...
    [sum_iae_tr_koop_all; sum_iae_tr_strejc_all], ...
    [sum_ise_tr_koop_all; sum_ise_tr_strejc_all], ...
    [sum_itae_tr_koop_all; sum_itae_tr_strejc_all], ...
    [sum_rmse_tr_koop_all; sum_rmse_tr_strejc_all], ...
    [sum_peak_koop_all; sum_peak_strejc_all], ...
    [sum_over_koop_all; sum_over_strejc_all], ...
    [sum_under_koop_all; sum_under_strejc_all], ...
    [sum_tvu_koop_all; sum_tvu_strejc_all], ...
    [sum_dumax_koop_all; sum_dumax_strejc_all], ...
    [sum_uenergy_koop_all; sum_uenergy_strejc_all], ...
    'VariableNames', {'Controller', ...
    'Total_RMSE','Total_IAE','Total_Objective_Scaled', ...
    'Settled_Cases','Total_SettlingTime', ...
    'Total_IAE_Transient','Total_ISE_Transient','Total_ITAE_Transient','Total_RMSE_Transient', ...
    'Total_PeakDeviation','Total_Overshoot','Total_Undershoot', ...
    'Total_TVu_Transient','Total_DeltaUMax_Transient','Total_ControlEnergy_Transient'} );

disp('===== FINAL SUM OVER ALL INITIAL CONDITIONS =====');
disp(metrics_total);

disp('===== PER-T0 METRICS =====');
disp(all_rows);

%% ===== AVERAGE TABLE =====
metrics_avg = metrics_total;

for r = 1:height(metrics_avg)
    settled_cases = metrics_avg.Settled_Cases(r);
    if settled_cases > 0
        metrics_avg.Total_SettlingTime(r) = metrics_avg.Total_SettlingTime(r) / settled_cases;
    else
        metrics_avg.Total_SettlingTime(r) = NaN;
    end
end

n_cases = length(temps);
cols_to_divide = setdiff(2:width(metrics_avg), 5); % keep Settled_Cases unchanged, SettlingTime handled above
metrics_avg{:, cols_to_divide} = metrics_avg{:, cols_to_divide} / n_cases;

disp('===== AVERAGE PER INITIAL CONDITION =====');
disp(metrics_avg);

%% ===== SAVE =====
writetable(all_rows, fullfile(res_dir, 'per_T0_metrics_dynamic.csv'));
writetable(metrics_total, fullfile(res_dir, 'metrics_total_dynamic.csv'));
writetable(metrics_avg, fullfile(res_dir, 'metrics_avg_dynamic.csv'));

disp('Saved:');
disp(fullfile(res_dir, 'per_T0_metrics_dynamic.csv'));
disp(fullfile(res_dir, 'metrics_total_dynamic.csv'));
disp(fullfile(res_dir, 'metrics_avg_dynamic.csv'));


%% ===== METRICS vs INITIAL CONDITION =====
plot_metric_figure(T0_vec, RMSE_K, RMSE_S, 'RMSE', 'RMSE ($^\circ$C)', ...
    fullfile(fig_dir, 'metric_RMSE.png'));

plot_metric_figure(T0_vec, IAE_K, IAE_S, 'IAE', 'IAE', ...
    fullfile(fig_dir, 'metric_IAE.png'));

plot_metric_figure(T0_vec, OBJ_K, OBJ_S, 'Objective', 'Objective scaled', ...
    fullfile(fig_dir, 'metric_Objective.png'));

plot_metric_figure(T0_vec, TS_K, TS_S, 'Settling time', 'Settling time (samples)', ...
    fullfile(fig_dir, 'metric_SettlingTime.png'));

plot_metric_figure(T0_vec, IAEtr_K, IAEtr_S, 'IAE transient', 'IAE transient', ...
    fullfile(fig_dir, 'metric_IAEtr.png'));

plot_metric_figure(T0_vec, ISEtr_K, ISEtr_S, 'ISE transient', 'ISE transient', ...
    fullfile(fig_dir, 'metric_ISEtr.png'));

plot_metric_figure(T0_vec, ITAEtr_K, ITAEtr_S, 'ITAE transient', 'ITAE transient', ...
    fullfile(fig_dir, 'metric_ITAEtr.png'));

plot_metric_figure(T0_vec, RMSEtr_K, RMSEtr_S, 'RMSE transient', 'RMSE transient', ...
    fullfile(fig_dir, 'metric_RMSEtr.png'));

plot_metric_figure(T0_vec, PEAK_K, PEAK_S, 'Peak deviation', 'Peak deviation ($^\circ$C)', ...
    fullfile(fig_dir, 'metric_Peak.png'));

plot_metric_figure(T0_vec, OVER_K, OVER_S, 'Overshoot', 'Overshoot ($^\circ$C)', ...
    fullfile(fig_dir, 'metric_Overshoot.png'));

plot_metric_figure(T0_vec, UNDER_K, UNDER_S, 'Undershoot', 'Undershoot ($^\circ$C)', ...
    fullfile(fig_dir, 'metric_Undershoot.png'));

plot_metric_figure(T0_vec, TVU_K, TVU_S, 'TVu transient', 'TVu', ...
    fullfile(fig_dir, 'metric_TVu.png'));

plot_metric_figure(T0_vec, DUMAX_K, DUMAX_S, 'Maximum input increment', 'Max $|\Delta u|$', ...
    fullfile(fig_dir, 'metric_DeltaUMax.png'));

plot_metric_figure(T0_vec, UEN_K, UEN_S, 'Control energy', 'Control energy', ...
    fullfile(fig_dir, 'metric_ControlEnergy.png'));

%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================

function [ts_idx, metrics] = compute_dynamic_transient_metrics(y, u, target, settling_band, min_hold, inside_fraction_required)
    y = y(:);
    u = u(:);
    N = min(length(y), length(u));

    y = y(1:N);
    u = u(1:N);

    e = y - target;
    abs_e = abs(e);

    inside = abs_e <= settling_band;
    ts_idx = NaN;

    for i = 1:(N - min_hold + 1)
        window = inside(i:i+min_hold-1);
        if mean(window) >= inside_fraction_required
            ts_idx = i;
            break;
        end
    end

    if isnan(ts_idx)
        idx_tr = 1:N;
        metrics.SettlingFound = false;
        metrics.SettlingIndex = NaN;
        metrics.SettlingTime  = NaN;
    else
        idx_tr = 1:ts_idx;
        metrics.SettlingFound = true;
        metrics.SettlingIndex = ts_idx;
        metrics.SettlingTime  = ts_idx - 1;
    end

    e_tr = e(idx_tr);
    y_tr = y(idx_tr);
    u_tr = u(idx_tr);

    metrics.TransientLength = length(idx_tr);

    metrics.IAE_transient  = sum(abs(e_tr));
    metrics.ISE_transient  = sum(e_tr.^2);
    metrics.ITAE_transient = sum((0:length(e_tr)-1)' .* abs(e_tr));
    metrics.RMSE_transient = sqrt(mean(e_tr.^2));

    [~, idx_peak] = max(abs(e_tr));
    metrics.PeakDeviation = abs(e_tr(idx_peak));
    metrics.PeakIndex     = idx_peak;

    metrics.Overshoot  = max(0, max(y_tr - target));
    metrics.Undershoot = max(0, max(target - y_tr));

    if length(u_tr) > 1
        du = diff(u_tr);
        metrics.TVu_transient = sum(abs(du));
        metrics.DeltaUMax     = max(abs(du));
    else
        metrics.TVu_transient = 0;
        metrics.DeltaUMax     = 0;
    end

    metrics.ControlEnergy = sum(u_tr.^2);
end

function s = num2str_or_nan(x)
    if isnan(x)
        s = 'NaN';
    else
        s = sprintf('%d', round(x));
    end
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