%% Auto-plot + transient metrics for selected start temperatures
% Only settling-based transient metrics
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
u_mean = 65.8447; %#ok<NASGU>

%% ===== SETTINGS =====
time_format = 'yyyy-MM-dd HH:mm:ss.SSS';

temps    = [55, 58, 60, 62, 66, 68];
file_ids = [3, 4, 5, 6, 7, 8];   % original file numbering

% target used in plots / evaluation
target = 59.0676 + abs(59.0676 - 63.8084);

%% ===== TRANSIENT SETTINGS =====
settling_band = 0.4;   % +/- band around target
min_hold      = 5;    % must stay inside for at least 10 samples

%% ===== GLOBAL ACCUMULATORS (TRANSIENT ONLY) =====
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

sum_uenergy_koop_all   = 0;
sum_uenergy_strejc_all = 0;

%% ===== TABLE STORAGE =====
all_rows = [];

%% ===== MAIN LOOP =====
for k = 1:length(temps)

    T0  = temps(k);
    fid = file_ids(k);

    fprintf('\n====================================================\n');
    fprintf('Processing start temperature T0 = %d °C (file id %d)\n', T0, fid);
    fprintf('====================================================\n');

    koop_file   = sprintf('steps/2611runtime_log_koop%d.mat', fid);
    strejc_file = sprintf('steps/2611runtime_log_strejc%d.mat', fid);

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

    %% ===== TRANSIENT METRICS: KOOPMAN =====
    e_koop      = T4_koop - target;
    abs_e_koop  = abs(e_koop);
    inside_koop = abs_e_koop <= settling_band;

    ts_idx_koop = L;   % fallback if never settles
    for i = 1:(L - min_hold + 1)
        if all(inside_koop(i:i+min_hold-1))
            ts_idx_koop = i;
            break;
        end
    end

    idx_tr_koop = 1:ts_idx_koop;
    e_tr_koop   = e_koop(idx_tr_koop);
    u_tr_koop   = Pump_koop(idx_tr_koop);

    koop_settling_index = ts_idx_koop;
    koop_settling_time  = ts_idx_koop - 1;
    koop_iae            = sum(abs(e_tr_koop));
    koop_ise            = sum(e_tr_koop.^2);
    koop_itae           = sum((0:length(e_tr_koop)-1)' .* abs(e_tr_koop));
    koop_rmse           = sqrt(mean(e_tr_koop.^2));
    koop_u_energy       = sum(u_tr_koop.^2);

    %% ===== TRANSIENT METRICS: STREJC =====
    e_strejc      = T4_strejc - target;
    abs_e_strejc  = abs(e_strejc);
    inside_strejc = abs_e_strejc <= settling_band;

    ts_idx_strejc = L;   % fallback if never settles
    for i = 1:(L - min_hold + 1)
        if all(inside_strejc(i:i+min_hold-1))
            ts_idx_strejc = i;
            break;
        end
    end

    idx_tr_strejc = 1:ts_idx_strejc;
    e_tr_strejc   = e_strejc(idx_tr_strejc);
    u_tr_strejc   = Pump_strejc(idx_tr_strejc);

    strejc_settling_index = ts_idx_strejc;
    strejc_settling_time  = ts_idx_strejc - 1;
    strejc_iae            = sum(abs(e_tr_strejc));
    strejc_ise            = sum(e_tr_strejc.^2);
    strejc_itae           = sum((0:length(e_tr_strejc)-1)' .* abs(e_tr_strejc));
    strejc_rmse           = sqrt(mean(e_tr_strejc.^2));
    strejc_u_energy       = sum(u_tr_strejc.^2);

    %% ===== ACCUMULATE TRANSIENT SUMS =====
    sum_ts_koop_all        = sum_ts_koop_all        + koop_settling_time;
    sum_ts_strejc_all      = sum_ts_strejc_all      + strejc_settling_time;

    sum_iae_tr_koop_all    = sum_iae_tr_koop_all    + koop_iae;
    sum_iae_tr_strejc_all  = sum_iae_tr_strejc_all  + strejc_iae;

    sum_ise_tr_koop_all    = sum_ise_tr_koop_all    + koop_ise;
    sum_ise_tr_strejc_all  = sum_ise_tr_strejc_all  + strejc_ise;

    sum_itae_tr_koop_all   = sum_itae_tr_koop_all   + koop_itae;
    sum_itae_tr_strejc_all = sum_itae_tr_strejc_all + strejc_itae;

    sum_rmse_tr_koop_all   = sum_rmse_tr_koop_all   + koop_rmse;
    sum_rmse_tr_strejc_all = sum_rmse_tr_strejc_all + strejc_rmse;

    sum_uenergy_koop_all   = sum_uenergy_koop_all   + koop_u_energy;
    sum_uenergy_strejc_all = sum_uenergy_strejc_all + strejc_u_energy;

    %% ===== PRINT PER-T0 SUMMARY =====
    fprintf('Koopman transient metrics:\n');
    fprintf('  Settling time  = %d samples\n', koop_settling_time);
    fprintf('  RMSE           = %.3f\n', koop_rmse);
    fprintf('  IAE            = %.3f\n', koop_iae);
    fprintf('  ISE            = %.3f\n', koop_ise);
    fprintf('  ITAE           = %.3f\n', koop_itae);
    fprintf('  Control energy = %.3f\n', koop_u_energy);

    fprintf('Strejc transient metrics:\n');
    fprintf('  Settling time  = %d samples\n', strejc_settling_time);
    fprintf('  RMSE           = %.3f\n', strejc_rmse);
    fprintf('  IAE            = %.3f\n', strejc_iae);
    fprintf('  ISE            = %.3f\n', strejc_ise);
    fprintf('  ITAE           = %.3f\n', strejc_itae);
    fprintf('  Control energy = %.3f\n', strejc_u_energy);

    %% ===== STORE INTO TABLE =====
    all_rows = [all_rows; ...
        table(T0, "Koopman", ...
              koop_settling_time, koop_settling_index, ...
              koop_rmse, koop_iae, koop_ise, koop_itae, koop_u_energy, ...
              'VariableNames', {'T0','Controller', ...
              'SettlingTime','SettlingIndex', ...
              'RMSE_Transient','IAE_Transient','ISE_Transient','ITAE_Transient','ControlEnergy_Transient'}); ...
        table(T0, "Strejc", ...
              strejc_settling_time, strejc_settling_index, ...
              strejc_rmse, strejc_iae, strejc_ise, strejc_itae, strejc_u_energy, ...
              'VariableNames', {'T0','Controller', ...
              'SettlingTime','SettlingIndex', ...
              'RMSE_Transient','IAE_Transient','ISE_Transient','ITAE_Transient','ControlEnergy_Transient'})];

    %% ===== PLOT =====
    figure('Color','w','Position',[100 100 900 520]);
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
    legend([h1 h2 h3], {'Koopman MPC','Linear MPC','Target'}, 'Location','best');

    ylim([min([T4_koop; T4_strejc; target])-1, max([T4_koop; T4_strejc; target])+1]);

    % --- INPUT ---
    nexttile;
    h4 = stairs(step, Pump_koop, 'm', 'LineWidth', 2); hold on;
    h5 = stairs(step, Pump_strejc, 'b--', 'LineWidth', 2);

    xline(ts_idx_koop-1, 'm:', 'LineWidth', 1.2, 'HandleVisibility','off');
    xline(ts_idx_strejc-1, 'b:', 'LineWidth', 1.2, 'HandleVisibility','off');

    grid on; grid minor;
    xlabel('Time step');
    ylabel('Pump speed (\%)');
    title('Control input');
    legend([h4 h5], {'Koopman MPC','Linear MPC'}, 'Location','best');

    out_png = fullfile(fig_dir, sprintf('compare_cl_T0_%d.png', T0));
    saveas(gcf, out_png);
end

%% ===== FINAL SUM TABLE (TRANSIENT ONLY) =====
metrics_sum = table( ...
    {'Koopman'; 'Strejc'}, ...
    [sum_ts_koop_all; sum_ts_strejc_all], ...
    [sum_rmse_tr_koop_all; sum_rmse_tr_strejc_all], ...
    [sum_iae_tr_koop_all; sum_iae_tr_strejc_all], ...
    [sum_ise_tr_koop_all; sum_ise_tr_strejc_all], ...
    [sum_itae_tr_koop_all; sum_itae_tr_strejc_all], ...
    [sum_uenergy_koop_all; sum_uenergy_strejc_all], ...
    'VariableNames', {'Controller', ...
    'Sum_SettlingTime','Sum_RMSE_Transient','Sum_IAE_Transient', ...
    'Sum_ISE_Transient','Sum_ITAE_Transient','Sum_ControlEnergy_Transient'});

%% ===== FINAL MEAN TABLE (TRANSIENT ONLY) =====
n_cases = length(temps);

metrics_mean = metrics_sum;
metrics_mean{:, 2:end} = metrics_mean{:, 2:end} / n_cases;

disp('===== PER-T0 TRANSIENT METRICS =====');
disp(all_rows);

disp('===== SUM OF TRANSIENT METRICS =====');
disp(metrics_sum);

disp('===== MEAN OF TRANSIENT METRICS =====');
disp(metrics_mean);

%% ===== SAVE TABLES =====
writetable(all_rows,      fullfile(res_dir, 'per_T0_transient_metrics.csv'));
writetable(metrics_sum,   fullfile(res_dir, 'transient_metrics_sum.csv'));
writetable(metrics_mean,  fullfile(res_dir, 'transient_metrics_mean.csv'));

disp('Saved:');
disp(' - results/per_T0_transient_metrics.csv');
disp(' - results/transient_metrics_sum.csv');
disp(' - results/transient_metrics_mean.csv');