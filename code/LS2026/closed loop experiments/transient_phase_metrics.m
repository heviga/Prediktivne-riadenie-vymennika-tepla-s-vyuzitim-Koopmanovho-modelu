%% Auto-plot + metrics for ALL start temperatures
% Full-response metrics + transient metrics
%% Auto-plot + metrics for ALL start temperatures
clc; clear; close all;

project_root = 'C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\LS2026';
addpath(genpath(project_root));

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');

%% ===== SCRIPT PATH / OUTPUT PATHS =====
script_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(script_dir, 'figs', 'transient');
res_dir = fullfile(script_dir, 'results');

if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

if ~exist(res_dir, 'dir')
    mkdir(res_dir);
end
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

% target used in your plots / evaluation
x_mean_target = 59.0676 + abs(59.0676 - 63.8084);

%% ===== TRANSIENT SETTINGS =====
settling_band = 0.4;   % +/- 0.5 degC band around target
min_hold      = 10;    % must stay inside for at least 10 samples
Ntr_fixed     = 100;   % fixed transient horizon (samples)


%% ===== GLOBAL ACCUMULATORS =====
sum_obj_koop_scaled_all   = 0;
sum_obj_strejc_scaled_all = 0;

sum_iae_koop_all   = 0;
sum_iae_strejc_all = 0;

sum_rmse_koop_all   = 0;
sum_rmse_strejc_all = 0;

% transient (settling-based)
sum_ts_koop_all        = 0;
sum_ts_strejc_all      = 0;
sum_iae_tr_koop_all    = 0;
sum_iae_tr_strejc_all  = 0;
sum_ise_tr_koop_all    = 0;
sum_ise_tr_strejc_all  = 0;
sum_itae_tr_koop_all   = 0;
sum_itae_tr_strejc_all = 0;
sum_peak_koop_all      = 0;
sum_peak_strejc_all    = 0;
sum_tvu_koop_all       = 0;
sum_tvu_strejc_all     = 0;
sum_uenergy_koop_all   = 0;
sum_uenergy_strejc_all = 0;

% transient (fixed horizon)
sum_iae_fix_koop_all    = 0;
sum_iae_fix_strejc_all  = 0;
sum_ise_fix_koop_all    = 0;
sum_ise_fix_strejc_all  = 0;
sum_itae_fix_koop_all   = 0;
sum_itae_fix_strejc_all = 0;
sum_rmse_fix_koop_all   = 0;
sum_rmse_fix_strejc_all = 0;
sum_tvu_fix_koop_all    = 0;
sum_tvu_fix_strejc_all  = 0;
sum_uen_fix_koop_all    = 0;
sum_uen_fix_strejc_all  = 0;

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

    t = datetime(string(log_data.timestamp), 'InputFormat', time_format);
    time_koop = seconds(t - t(1)); %#ok<NASGU>
    T4_koop   = log_data.T4(:);
    Pump_koop = log_data.Pump2(:);

    %% ===== LOAD STREJC =====
    S = load(strejc_file);
    log_data = S.log_data;

    t = datetime(string(log_data.timestamp), 'InputFormat', time_format);
    time_strejc = seconds(t - t(1)); %#ok<NASGU>
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

    % Objective physical
    obj_koop   = sum(Q_cost*(T4_koop   - target).^2 + R_cost*(Pump_koop).^2);
    obj_strejc = sum(Q_cost*(T4_strejc - target).^2 + R_cost*(Pump_strejc).^2);

    % Objective scaled
    obj_koop_scaled = sum( ...
        Q_cost*((T4_koop   - bias - x_mean)/x_std).^2 + ...
        R_cost*((Pump_koop - u_mean)/u_std).^2 );

    obj_strejc_scaled = sum( ...
        Q_cost*((T4_strejc - bias - x_mean)/x_std).^2 + ...
        R_cost*((Pump_strejc - u_mean)/u_std).^2 );

    sum_obj_koop_scaled_all   = sum_obj_koop_scaled_all   + obj_koop_scaled;
    sum_obj_strejc_scaled_all = sum_obj_strejc_scaled_all + obj_strejc_scaled;

    %% ===== TRANSIENT METRICS: SETTLING-BASED =====
    [ts_idx_koop, tr_koop] = compute_transient_metrics(T4_koop, Pump_koop, target, settling_band, min_hold);
    [ts_idx_strejc, tr_strejc] = compute_transient_metrics(T4_strejc, Pump_strejc, target, settling_band, min_hold);

    sum_ts_koop_all        = sum_ts_koop_all        + tr_koop.SettlingTime;
    sum_ts_strejc_all      = sum_ts_strejc_all      + tr_strejc.SettlingTime;
    sum_iae_tr_koop_all    = sum_iae_tr_koop_all    + tr_koop.IAE_transient;
    sum_iae_tr_strejc_all  = sum_iae_tr_strejc_all  + tr_strejc.IAE_transient;
    sum_ise_tr_koop_all    = sum_ise_tr_koop_all    + tr_koop.ISE_transient;
    sum_ise_tr_strejc_all  = sum_ise_tr_strejc_all  + tr_strejc.ISE_transient;
    sum_itae_tr_koop_all   = sum_itae_tr_koop_all   + tr_koop.ITAE_transient;
    sum_itae_tr_strejc_all = sum_itae_tr_strejc_all + tr_strejc.ITAE_transient;
    sum_peak_koop_all      = sum_peak_koop_all      + tr_koop.PeakDeviation;
    sum_peak_strejc_all    = sum_peak_strejc_all    + tr_strejc.PeakDeviation;
    sum_tvu_koop_all       = sum_tvu_koop_all       + tr_koop.TVu_transient;
    sum_tvu_strejc_all     = sum_tvu_strejc_all     + tr_strejc.TVu_transient;
    sum_uenergy_koop_all   = sum_uenergy_koop_all   + tr_koop.ControlEnergy;
    sum_uenergy_strejc_all = sum_uenergy_strejc_all + tr_strejc.ControlEnergy;

    %% ===== TRANSIENT METRICS: FIXED HORIZON =====
    fix_koop   = compute_fixed_horizon_metrics(T4_koop, Pump_koop, target, Ntr_fixed);
    fix_strejc = compute_fixed_horizon_metrics(T4_strejc, Pump_strejc, target, Ntr_fixed);

    sum_iae_fix_koop_all    = sum_iae_fix_koop_all    + fix_koop.IAE;
    sum_iae_fix_strejc_all  = sum_iae_fix_strejc_all  + fix_strejc.IAE;
    sum_ise_fix_koop_all    = sum_ise_fix_koop_all    + fix_koop.ISE;
    sum_ise_fix_strejc_all  = sum_ise_fix_strejc_all  + fix_strejc.ISE;
    sum_itae_fix_koop_all   = sum_itae_fix_koop_all   + fix_koop.ITAE;
    sum_itae_fix_strejc_all = sum_itae_fix_strejc_all + fix_strejc.ITAE;
    sum_rmse_fix_koop_all   = sum_rmse_fix_koop_all   + fix_koop.RMSE;
    sum_rmse_fix_strejc_all = sum_rmse_fix_strejc_all + fix_strejc.RMSE;
    sum_tvu_fix_koop_all    = sum_tvu_fix_koop_all    + fix_koop.TVu;
    sum_tvu_fix_strejc_all  = sum_tvu_fix_strejc_all  + fix_strejc.TVu;
    sum_uen_fix_koop_all    = sum_uen_fix_koop_all    + fix_koop.ControlEnergy;
    sum_uen_fix_strejc_all  = sum_uen_fix_strejc_all  + fix_strejc.ControlEnergy;

    %% ===== PRINT PER-T0 SUMMARY =====
    fprintf(['T0=%d | FULL: Koop RMSE=%.3f IAE=%.1f ObjS=%.2f | Strejc RMSE=%.3f IAE=%.1f ObjS=%.2f\n'], ...
        T0, rmse_koop, iae_koop, obj_koop_scaled, rmse_strejc, iae_strejc, obj_strejc_scaled);

    fprintf(['T0=%d | SETTLING TRANSIENT: Koop Ts=%d IAEtr=%.2f ITAEtr=%.2f Peak=%.2f TVu=%.2f | ' ...
             'Strejc Ts=%d IAEtr=%.2f ITAEtr=%.2f Peak=%.2f TVu=%.2f\n'], ...
        T0, ...
        tr_koop.SettlingTime, tr_koop.IAE_transient, tr_koop.ITAE_transient, tr_koop.PeakDeviation, tr_koop.TVu_transient, ...
        tr_strejc.SettlingTime, tr_strejc.IAE_transient, tr_strejc.ITAE_transient, tr_strejc.PeakDeviation, tr_strejc.TVu_transient);

    fprintf(['T0=%d | FIXED %d STEPS: Koop RMSE=%.3f IAE=%.2f ITAE=%.2f TVu=%.2f | ' ...
             'Strejc RMSE=%.3f IAE=%.2f ITAE=%.2f TVu=%.2f\n'], ...
        T0, Ntr_fixed, ...
        fix_koop.RMSE, fix_koop.IAE, fix_koop.ITAE, fix_koop.TVu, ...
        fix_strejc.RMSE, fix_strejc.IAE, fix_strejc.ITAE, fix_strejc.TVu);

    %% ===== STORE INTO TABLE =====
    all_rows = [all_rows; ...
        table(T0, "Koopman", ...
              rmse_koop, iae_koop, obj_koop, obj_koop_scaled, ...
              tr_koop.SettlingTime, tr_koop.SettlingIndex, ...
              tr_koop.IAE_transient, tr_koop.ISE_transient, tr_koop.ITAE_transient, tr_koop.RMSE_transient, ...
              tr_koop.PeakDeviation, tr_koop.Overshoot, tr_koop.Undershoot, ...
              tr_koop.TVu_transient, tr_koop.DeltaUMax, tr_koop.ControlEnergy, ...
              fix_koop.N, fix_koop.IAE, fix_koop.ISE, fix_koop.ITAE, fix_koop.RMSE, ...
              fix_koop.PeakDeviation, fix_koop.TVu, fix_koop.DeltaUMax, fix_koop.ControlEnergy, ...
              'VariableNames', {'T0','Controller', ...
              'RMSE_T4','IAE','Objective','Objective_Scaled', ...
              'SettlingTime','SettlingIndex', ...
              'IAE_Transient','ISE_Transient','ITAE_Transient','RMSE_Transient', ...
              'PeakDeviation','Overshoot','Undershoot', ...
              'TVu_Transient','DeltaUMax_Transient','ControlEnergy_Transient', ...
              'FixedHorizon','IAE_Fixed','ISE_Fixed','ITAE_Fixed','RMSE_Fixed', ...
              'PeakDeviation_Fixed','TVu_Fixed','DeltaUMax_Fixed','ControlEnergy_Fixed'}); ...
        table(T0, "Strejc", ...
              rmse_strejc, iae_strejc, obj_strejc, obj_strejc_scaled, ...
              tr_strejc.SettlingTime, tr_strejc.SettlingIndex, ...
              tr_strejc.IAE_transient, tr_strejc.ISE_transient, tr_strejc.ITAE_transient, tr_strejc.RMSE_transient, ...
              tr_strejc.PeakDeviation, tr_strejc.Overshoot, tr_strejc.Undershoot, ...
              tr_strejc.TVu_transient, tr_strejc.DeltaUMax, tr_strejc.ControlEnergy, ...
              fix_strejc.N, fix_strejc.IAE, fix_strejc.ISE, fix_strejc.ITAE, fix_strejc.RMSE, ...
              fix_strejc.PeakDeviation, fix_strejc.TVu, fix_strejc.DeltaUMax, fix_strejc.ControlEnergy, ...
              'VariableNames', {'T0','Controller', ...
              'RMSE_T4','IAE','Objective','Objective_Scaled', ...
              'SettlingTime','SettlingIndex', ...
              'IAE_Transient','ISE_Transient','ITAE_Transient','RMSE_Transient', ...
              'PeakDeviation','Overshoot','Undershoot', ...
              'TVu_Transient','DeltaUMax_Transient','ControlEnergy_Transient', ...
              'FixedHorizon','IAE_Fixed','ISE_Fixed','ITAE_Fixed','RMSE_Fixed', ...
              'PeakDeviation_Fixed','TVu_Fixed','DeltaUMax_Fixed','ControlEnergy_Fixed'})];

 
    %% ===== PLOT =====
    fig = figure('Color','w','Position',[100 100 900 520]);
    tiledlayout(2,1,'TileSpacing','Compact','Padding','Compact');
    
    % --- OUTPUT ---
    nexttile;
    h1 = plot(step, T4_koop, 'm', 'LineWidth', 2); hold on;
    h2 = plot(step, T4_strejc, 'b--', 'LineWidth', 2);
    h3 = yline(target, 'k-', 'LineWidth', 1.2);
    
    yline(target + settling_band, 'k:', 'LineWidth', 0.8, 'HandleVisibility','off');
    yline(target - settling_band, 'k:', 'LineWidth', 0.8, 'HandleVisibility','off');
    
    xline(ts_idx_koop-1, 'm:', 'LineWidth', 1.2, 'HandleVisibility','off');
    xline(ts_idx_strejc-1, 'b:', 'LineWidth', 1.2, 'HandleVisibility','off');
    
    grid on; grid minor;
    ylabel('Outlet temperature ($^\circ$C)');
    title(sprintf('Closed-loop response (start %d$^\\circ$C)', T0));
    legend([h1 h2 h3], {'Koopman MPC','Strejc MPC','Target'}, 'Location','best');
    
    ylim([min([T4_koop; T4_strejc; target])-1, max([T4_koop; T4_strejc; target])+1]);
    
    % --- INPUT ---
    nexttile;
    h4 = stairs(step, Pump_koop, 'm', 'LineWidth', 2); hold on;
    h5 = stairs(step, Pump_strejc, 'b--', 'LineWidth', 2);
    %h6 = yline(u_mean, 'k-', 'LineWidth', 1.2);
    
    xline(ts_idx_koop-1, 'm:', 'LineWidth', 1.2, 'HandleVisibility','off');
    xline(ts_idx_strejc-1, 'b:', 'LineWidth', 1.2, 'HandleVisibility','off');
    
    grid on; grid minor;
    xlabel('Time step');
    ylabel('Pump speed (\%)');
    title('Control input');
    legend([h4 h5], {'Koopman MPC','Strejc MPC'}, 'Location','best');
    
    out_png = fullfile(fig_dir, sprintf('compare_cl_T0_%d.png', T0));
    saveas(fig, out_png);
end

%% ===== FINAL TOTAL TABLE =====
metrics_total = table( ...
    {'Koopman'; 'Strejc'}, ...
    [sum_rmse_koop_all; sum_rmse_strejc_all], ...
    [sum_iae_koop_all;  sum_iae_strejc_all], ...
    [sum_obj_koop_scaled_all; sum_obj_strejc_scaled_all], ...
    [sum_ts_koop_all; sum_ts_strejc_all], ...
    [sum_iae_tr_koop_all; sum_iae_tr_strejc_all], ...
    [sum_ise_tr_koop_all; sum_ise_tr_strejc_all], ...
    [sum_itae_tr_koop_all; sum_itae_tr_strejc_all], ...
    [sum_peak_koop_all; sum_peak_strejc_all], ...
    [sum_tvu_koop_all; sum_tvu_strejc_all], ...
    [sum_uenergy_koop_all; sum_uenergy_strejc_all], ...
    [sum_rmse_fix_koop_all; sum_rmse_fix_strejc_all], ...
    [sum_iae_fix_koop_all; sum_iae_fix_strejc_all], ...
    [sum_ise_fix_koop_all; sum_ise_fix_strejc_all], ...
    [sum_itae_fix_koop_all; sum_itae_fix_strejc_all], ...
    [sum_tvu_fix_koop_all; sum_tvu_fix_strejc_all], ...
    [sum_uen_fix_koop_all; sum_uen_fix_strejc_all], ...
    'VariableNames', {'Controller', ...
    'Total_RMSE','Total_IAE','Total_Objective_Scaled', ...
    'Total_SettlingTime','Total_IAE_Transient','Total_ISE_Transient','Total_ITAE_Transient', ...
    'Total_PeakDeviation','Total_TVu_Transient','Total_ControlEnergy_Transient', ...
    'Total_RMSE_Fixed','Total_IAE_Fixed','Total_ISE_Fixed','Total_ITAE_Fixed', ...
    'Total_TVu_Fixed','Total_ControlEnergy_Fixed'} );

disp('===== FINAL SUM OVER ALL INITIAL CONDITIONS =====');
disp(metrics_total);

disp('===== PER-T0 METRICS (Koopman + Strejc) =====');
disp(all_rows);

%% ===== OPTIONAL AVERAGE TABLE =====
n_cases = length(temps);

metrics_avg = metrics_total;
metrics_avg{:, 2:end} = metrics_avg{:, 2:end} / n_cases;

disp('===== AVERAGE PER INITIAL CONDITION =====');
disp(metrics_avg);

%% ===== SAVE TABLES =====
writetable(all_rows, fullfile(res_dir, 'per_T0_metrics.csv'));
writetable(metrics_total, fullfile(res_dir, 'metrics_total.csv'));
writetable(metrics_avg, fullfile(res_dir, 'metrics_avg.csv'));

disp('Saved:');
disp(' - results/per_T0_metrics.csv');
disp(' - results/metrics_total.csv');
disp(' - results/metrics_avg.csv');

%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================

function [ts_idx, metrics] = compute_transient_metrics(y, u, target, settling_band, min_hold)
    y = y(:);
    u = u(:);
    N = length(y);

    e = y - target;
    abs_e = abs(e);

    % first index where signal enters band and stays there for min_hold samples
    inside = abs_e <= settling_band;
    ts_idx = N;   % fallback if never settles

    for i = 1:(N - min_hold + 1)
        if all(inside(i:i+min_hold-1))
            ts_idx = i;
            break;
        end
    end

    idx_tr = 1:ts_idx;

    e_tr = e(idx_tr);
    y_tr = y(idx_tr);
    u_tr = u(idx_tr);

    metrics.SettlingIndex = ts_idx;
    metrics.SettlingTime  = ts_idx - 1;   % in samples

    metrics.IAE_transient  = sum(abs(e_tr));
    metrics.ISE_transient  = sum(e_tr.^2);
    metrics.ITAE_transient = sum((0:length(e_tr)-1)' .* abs(e_tr));
    metrics.RMSE_transient = sqrt(mean(e_tr.^2));

    metrics.MaxAbsError = max(abs(e_tr));

    metrics.Overshoot  = max(y_tr - target);
    metrics.Undershoot = max(target - y_tr);

    [~, idx_peak] = max(abs(e_tr));
    metrics.PeakDeviation = abs(e_tr(idx_peak));
    metrics.PeakIndex     = idx_peak;

    metrics.ControlEnergy = sum(u_tr.^2);
    metrics.ControlMean   = mean(u_tr);
    metrics.ControlMax    = max(u_tr);
    metrics.ControlMin    = min(u_tr);

    if length(u_tr) > 1
        du = diff(u_tr);
        metrics.TVu_transient = sum(abs(du));
        metrics.DeltaUMax     = max(abs(du));
    else
        metrics.TVu_transient = 0;
        metrics.DeltaUMax     = 0;
    end
end

function metrics = compute_fixed_horizon_metrics(y, u, target, Ntr)
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