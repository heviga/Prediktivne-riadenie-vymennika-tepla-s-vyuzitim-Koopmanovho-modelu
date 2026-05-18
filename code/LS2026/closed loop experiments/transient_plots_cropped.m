%% Auto-plot + transient metrics for selected start temperatures
% Settling-based transient metrics + transient objective function
% Plots are cropped to the later settling line + 1 sample.
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

temps    = [55, 58, 60, 62, 66, 68];
file_ids = [3, 4, 5, 6, 7, 8];

% target used in plots / evaluation
target = 59.0676 + abs(59.0676 - 63.8084);

Q_cost = 10;
R_cost = 1;

%% ===== TRANSIENT SETTINGS =====
settling_band = 0.4;
min_hold      = 5;

%% ===== GLOBAL ACCUMULATORS (TRANSIENT ONLY) =====
sum_ts_koop_all        = 0;
sum_ts_linear_all      = 0;

sum_iae_tr_koop_all    = 0;
sum_iae_tr_linear_all  = 0;

sum_ise_tr_koop_all    = 0;
sum_ise_tr_linear_all  = 0;

sum_itae_tr_koop_all   = 0;
sum_itae_tr_linear_all = 0;

sum_rmse_tr_koop_all   = 0;
sum_rmse_tr_linear_all = 0;

sum_uenergy_koop_all   = 0;
sum_uenergy_linear_all = 0;

sum_jcl_koop_all       = 0;
sum_jcl_linear_all     = 0;

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
    linear_file = sprintf('steps/2611runtime_log_strejc%d.mat', fid);

    %% ===== LOAD KOOPMAN =====
    S = load(koop_file);
    log_data = S.log_data;

    t = datetime(string(log_data.timestamp), 'InputFormat', time_format);
    time_koop = seconds(t - t(1)); %#ok<NASGU>
    T4_koop   = log_data.T4(:);
    Pump_koop = log_data.Pump2(:);

    %% ===== LOAD LINEAR MODEL =====
    S = load(linear_file);
    log_data = S.log_data;

    t = datetime(string(log_data.timestamp), 'InputFormat', time_format);
    time_linear = seconds(t - t(1)); %#ok<NASGU>
    T4_linear   = log_data.T4(:);
    Pump_linear = log_data.Pump2(:);

    %% ===== LENGTHS FOR METRICS AND PLOTS =====
    % Metrics are evaluated over the common available interval.
    L_metric = min([length(T4_koop), length(T4_linear), length(Pump_koop), length(Pump_linear)]);

    % Full signals for plotting
    L_koop_plot   = min(length(T4_koop), length(Pump_koop));
    L_linear_plot = min(length(T4_linear), length(Pump_linear));

    T4_koop_plot     = T4_koop(1:L_koop_plot);
    Pump_koop_plot   = Pump_koop(1:L_koop_plot);
    step_koop_plot   = (0:L_koop_plot-1)';

    T4_linear_plot   = T4_linear(1:L_linear_plot);
    Pump_linear_plot = Pump_linear(1:L_linear_plot);
    step_linear_plot = (0:L_linear_plot-1)';

    % Common signals for metrics
    T4_koop     = T4_koop(1:L_metric);
    T4_linear   = T4_linear(1:L_metric);
    Pump_koop   = Pump_koop(1:L_metric);
    Pump_linear = Pump_linear(1:L_metric);

    L = L_metric;

    %% ===== TRANSIENT METRICS: KOOPMAN =====
    e_koop      = T4_koop - target;
    abs_e_koop  = abs(e_koop);
    inside_koop = abs_e_koop <= settling_band;

    ts_idx_koop = L;
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

    %% ===== TRANSIENT METRICS: LINEAR MODEL =====
    e_linear      = T4_linear - target;
    abs_e_linear  = abs(e_linear);
    inside_linear = abs_e_linear <= settling_band;

    ts_idx_linear = L;
    for i = 1:(L - min_hold + 1)
        if all(inside_linear(i:i+min_hold-1))
            ts_idx_linear = i;
            break;
        end
    end

    idx_tr_linear = 1:ts_idx_linear;
    e_tr_linear   = e_linear(idx_tr_linear);
    u_tr_linear   = Pump_linear(idx_tr_linear);

    linear_settling_index = ts_idx_linear;
    linear_settling_time  = ts_idx_linear - 1;
    linear_iae            = sum(abs(e_tr_linear));
    linear_ise            = sum(e_tr_linear.^2);
    linear_itae           = sum((0:length(e_tr_linear)-1)' .* abs(e_tr_linear));
    linear_rmse           = sqrt(mean(e_tr_linear.^2));
    linear_u_energy       = sum(u_tr_linear.^2);

    %% ===== TRANSIENT OBJECTIVE FUNCTION =====
    bias = target - x_mean;

    koop_jcl = sum( ...
        Q_cost * ((T4_koop(idx_tr_koop) - bias - x_mean) / x_std).^2 + ...
        R_cost * ((Pump_koop(idx_tr_koop) - u_mean) / u_std).^2 );

    linear_jcl = sum( ...
        Q_cost * ((T4_linear(idx_tr_linear) - bias - x_mean) / x_std).^2 + ...
        R_cost * ((Pump_linear(idx_tr_linear) - u_mean) / u_std).^2 );

    %% ===== ACCUMULATE TRANSIENT SUMS =====
    sum_ts_koop_all        = sum_ts_koop_all        + koop_settling_time;
    sum_ts_linear_all      = sum_ts_linear_all      + linear_settling_time;

    sum_iae_tr_koop_all    = sum_iae_tr_koop_all    + koop_iae;
    sum_iae_tr_linear_all  = sum_iae_tr_linear_all  + linear_iae;

    sum_ise_tr_koop_all    = sum_ise_tr_koop_all    + koop_ise;
    sum_ise_tr_linear_all  = sum_ise_tr_linear_all  + linear_ise;

    sum_itae_tr_koop_all   = sum_itae_tr_koop_all   + koop_itae;
    sum_itae_tr_linear_all = sum_itae_tr_linear_all + linear_itae;

    sum_rmse_tr_koop_all   = sum_rmse_tr_koop_all   + koop_rmse;
    sum_rmse_tr_linear_all = sum_rmse_tr_linear_all + linear_rmse;

    sum_uenergy_koop_all   = sum_uenergy_koop_all   + koop_u_energy;
    sum_uenergy_linear_all = sum_uenergy_linear_all + linear_u_energy;

    sum_jcl_koop_all       = sum_jcl_koop_all       + koop_jcl;
    sum_jcl_linear_all     = sum_jcl_linear_all     + linear_jcl;

    %% ===== PRINT PER-T0 SUMMARY =====
    fprintf('Koopman transient metrics:\n');
    fprintf('  Settling time  = %d samples\n', koop_settling_time);
    fprintf('  RMSE           = %.3f\n', koop_rmse);
    fprintf('  IAE            = %.3f\n', koop_iae);
    fprintf('  ISE            = %.3f\n', koop_ise);
    fprintf('  ITAE           = %.3f\n', koop_itae);
    fprintf('  Control energy = %.3f\n', koop_u_energy);
    fprintf('  Objective Jcl  = %.3f\n', koop_jcl);

    fprintf('Linear transient metrics:\n');
    fprintf('  Settling time  = %d samples\n', linear_settling_time);
    fprintf('  RMSE           = %.3f\n', linear_rmse);
    fprintf('  IAE            = %.3f\n', linear_iae);
    fprintf('  ISE            = %.3f\n', linear_ise);
    fprintf('  ITAE           = %.3f\n', linear_itae);
    fprintf('  Control energy = %.3f\n', linear_u_energy);
    fprintf('  Objective Jcl  = %.3f\n', linear_jcl);

    %% ===== STORE INTO TABLE =====
    all_rows = [all_rows; ...
        table(T0, "Koopman", ...
              koop_settling_time, koop_settling_index, ...
              koop_rmse, koop_iae, koop_ise, koop_itae, ...
              koop_u_energy, koop_jcl, ...
              'VariableNames', {'T0','Controller', ...
              'SettlingTime','SettlingIndex', ...
              'RMSE_Transient','IAE_Transient','ISE_Transient','ITAE_Transient', ...
              'ControlEnergy_Transient','Jcl_Transient'}); ...
        table(T0, "Linear", ...
              linear_settling_time, linear_settling_index, ...
              linear_rmse, linear_iae, linear_ise, linear_itae, ...
              linear_u_energy, linear_jcl, ...
              'VariableNames', {'T0','Controller', ...
              'SettlingTime','SettlingIndex', ...
              'RMSE_Transient','IAE_Transient','ISE_Transient','ITAE_Transient', ...
              'ControlEnergy_Transient','Jcl_Transient'})];

    %% ===== PLOT: CROPPED TO LATER SETTLING LINE =====
    fig = figure('Color','w');
    tiledlayout(2,1,'TileSpacing','Compact','Padding','Compact');

    label_fs = 12;
    title_fs = 12;
    tick_fs  = 8;

    % x-axis ends one sample after the later settling line
    x_end = max([ts_idx_koop-1, ts_idx_linear-1]) + 1;

    % do not exceed available plotted signal lengths
    max_available_x = max([step_koop_plot(end), step_linear_plot(end)]);
    x_end = min(x_end, max_available_x);

    % --- OUTPUT ---
    nexttile;
    h1 = plot(step_koop_plot, T4_koop_plot, 'm', 'LineWidth', 1.5); hold on;
    h2 = plot(step_linear_plot, T4_linear_plot, 'b', 'LineWidth', 1.5);
    h3 = yline(target, 'k-', 'LineWidth', 1.2);

    yline(target + settling_band, 'k:', 'LineWidth', 0.8, 'HandleVisibility','off');
    yline(target - settling_band, 'k:', 'LineWidth', 0.8, 'HandleVisibility','off');

    xline(ts_idx_koop-1, 'm:', 'LineWidth', 1.2, 'HandleVisibility','off');
    xline(ts_idx_linear-1, 'b:', 'LineWidth', 1.2, 'HandleVisibility','off');

    grid on; grid minor;
    ylabel('Outlet temperature ($^\circ$C)', 'FontSize', label_fs);
    title(sprintf('Closed-loop response (start %d$^\\circ$C)', T0), 'FontSize', title_fs);
    legend([h1 h2 h3], {'Koopman MPC','Linear MPC','Target'}, 'Location','best');

    ax1 = gca;
    ax1.FontSize = tick_fs;

    xlim([0 x_end]);

    % y-limits over visible part only
    idx_koop_vis   = step_koop_plot <= x_end;
    idx_linear_vis = step_linear_plot <= x_end;

    visible_y = [T4_koop_plot(idx_koop_vis); T4_linear_plot(idx_linear_vis); target];
    ylim([min(visible_y)-1, max(visible_y)+1]);

    % --- INPUT ---
    nexttile;
    h4 = stairs(step_koop_plot, Pump_koop_plot, 'm', 'LineWidth', 1.5); hold on;
    h5 = stairs(step_linear_plot, Pump_linear_plot, 'b', 'LineWidth', 1.5);

    xline(ts_idx_koop-1, 'm:', 'LineWidth', 1.2, 'HandleVisibility','off');
    xline(ts_idx_linear-1, 'b:', 'LineWidth', 1.2, 'HandleVisibility','off');

    grid on; grid minor;
    xlabel('Time step', 'FontSize', label_fs);
    ylabel('Pump speed (\%)', 'FontSize', label_fs);
    title('Control input', 'FontSize', title_fs);
    legend([h4 h5], {'Koopman MPC','Linear MPC'}, 'Location','best');

    xlim([0 x_end]);
    ylim([55 101.5]);

    ax2 = gca;
    ax2.FontSize = tick_fs;

    out_png = fullfile(fig_dir, sprintf('compare_cl_T0_%d_cropped_settling.png', T0));
    saveas(fig, out_png);

end

%% ===== FINAL SUM TABLE (TRANSIENT ONLY) =====
metrics_sum = table( ...
    {'Koopman'; 'Linear'}, ...
    [sum_ts_koop_all; sum_ts_linear_all], ...
    [sum_rmse_tr_koop_all; sum_rmse_tr_linear_all], ...
    [sum_iae_tr_koop_all; sum_iae_tr_linear_all], ...
    [sum_ise_tr_koop_all; sum_ise_tr_linear_all], ...
    [sum_itae_tr_koop_all; sum_itae_tr_linear_all], ...
    [sum_uenergy_koop_all; sum_uenergy_linear_all], ...
    [sum_jcl_koop_all; sum_jcl_linear_all], ...
    'VariableNames', {'Controller', ...
    'Sum_SettlingTime','Sum_RMSE_Transient','Sum_IAE_Transient', ...
    'Sum_ISE_Transient','Sum_ITAE_Transient','Sum_ControlEnergy_Transient', ...
    'Sum_Jcl_Transient'});

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

%% ===== SUMMARY PLOTS OF TRANSIENT METRICS VS INITIAL TEMPERATURE =====
koop_rows   = strcmp(all_rows.Controller, "Koopman");
linear_rows = strcmp(all_rows.Controller, "Linear");

koop_tbl   = sortrows(all_rows(koop_rows, :), 'T0');
linear_tbl = sortrows(all_rows(linear_rows, :), 'T0');

T0_plot = koop_tbl.T0;

Ts_koop   = koop_tbl.SettlingTime;
Ts_linear = linear_tbl.SettlingTime;

IAE_koop   = koop_tbl.IAE_Transient;
IAE_linear = linear_tbl.IAE_Transient;

ISE_koop   = koop_tbl.ISE_Transient;
ISE_linear = linear_tbl.ISE_Transient;

ITAE_koop   = koop_tbl.ITAE_Transient;
ITAE_linear = linear_tbl.ITAE_Transient;

Jcl_koop   = koop_tbl.Jcl_Transient;
Jcl_linear = linear_tbl.Jcl_Transient;

% --- style ---
label_fs  = 22;
title_fs  = 22;
tick_fs   = 12;
legend_fs = 12;

lw = 2.2;
ms = 8;

%% ===== FIGURE 1: Ts, IAE, ISE =====
fig_sum1 = figure('Color','w','Position',[100 100 1500 500]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

% Settling time
nexttile;
plot(T0_plot, Ts_koop,   'm-o', 'LineWidth', lw, 'MarkerSize', ms); hold on;
plot(T0_plot, Ts_linear, 'b-s', 'LineWidth', lw, 'MarkerSize', ms);
grid on; box on; grid minor;
xlim([55 max(T0_plot)]);

ax = gca;
ax.FontSize = tick_fs;

title('Settling time', 'FontSize', title_fs, 'Interpreter','latex');
xlabel('Initial temperature $T_0$ ($^\circ$C)', 'FontSize', label_fs, 'Interpreter','latex');
ylabel('$T_s$ (samples)', 'FontSize', label_fs, 'Interpreter','latex');
legend('Koopman','Linear','Location','northeast','FontSize',legend_fs,'Interpreter','latex');

% IAE
nexttile;
plot(T0_plot, IAE_koop,   'm-o', 'LineWidth', lw, 'MarkerSize', ms); hold on;
plot(T0_plot, IAE_linear, 'b-s', 'LineWidth', lw, 'MarkerSize', ms);
grid on; box on; grid minor;
xlim([55 max(T0_plot)]);

ax = gca;
ax.FontSize = tick_fs;

title('IAE', 'FontSize', title_fs, 'Interpreter','latex');
xlabel('Initial temperature $T_0$ ($^\circ$C)', 'FontSize', label_fs, 'Interpreter','latex');
ylabel('IAE', 'FontSize', label_fs, 'Interpreter','latex');
legend('Koopman','Linear','Location','northeast','FontSize',legend_fs,'Interpreter','latex');

% ISE
nexttile;
plot(T0_plot, ISE_koop,   'm-o', 'LineWidth', lw, 'MarkerSize', ms); hold on;
plot(T0_plot, ISE_linear, 'b-s', 'LineWidth', lw, 'MarkerSize', ms);
grid on; box on; grid minor;
xlim([55 max(T0_plot)]);

ax = gca;
ax.FontSize = tick_fs;

title('ISE', 'FontSize', title_fs, 'Interpreter','latex');
xlabel('Initial temperature $T_0$ ($^\circ$C)', 'FontSize', label_fs, 'Interpreter','latex');
ylabel('ISE', 'FontSize', label_fs, 'Interpreter','latex');
legend('Koopman','Linear','Location','northeast','FontSize',legend_fs,'Interpreter','latex');

out_pdf1 = fullfile(fig_dir, 'transient_metrics_summary_1.pdf');
out_png1 = fullfile(fig_dir, 'transient_metrics_summary_1.png');

exportgraphics(fig_sum1, out_pdf1, 'ContentType','vector');
exportgraphics(fig_sum1, out_png1, 'Resolution', 300);

%% ===== FIGURE 2: ITAE, JCL =====
fig_sum2 = figure('Color','w','Position',[100 100 1500 500]);
tiledlayout(1,6,'TileSpacing','compact','Padding','compact');

% ITAE
nexttile(2,[1 2]);
plot(T0_plot, ITAE_koop,   'm-o', 'LineWidth', lw, 'MarkerSize', ms); hold on;
plot(T0_plot, ITAE_linear, 'b-s', 'LineWidth', lw, 'MarkerSize', ms);
grid on; box on; grid minor;
xlim([55 max(T0_plot)]);

ax = gca;
ax.FontSize = tick_fs;

title('ITAE', 'FontSize', title_fs, 'Interpreter','latex');
xlabel('Initial temperature $T_0$ ($^\circ$C)', 'FontSize', label_fs, 'Interpreter','latex');
ylabel('ITAE', 'FontSize', label_fs, 'Interpreter','latex');
legend('Koopman','Linear','Location','northeast','FontSize',legend_fs,'Interpreter','latex');

% Objective value
nexttile(4,[1 2]);
plot(T0_plot, Jcl_koop,   'm-o', 'LineWidth', lw, 'MarkerSize', ms); hold on;
plot(T0_plot, Jcl_linear, 'b-s', 'LineWidth', lw, 'MarkerSize', ms);
grid on; box on; grid minor;
xlim([55 max(T0_plot)]);

ax = gca;
ax.FontSize = tick_fs;

title('Objective value', 'FontSize', title_fs, 'Interpreter','latex');
xlabel('Initial temperature $T_0$ ($^\circ$C)', 'FontSize', label_fs, 'Interpreter','latex');
ylabel('$J_{\mathrm{CL}}$', 'FontSize', label_fs, 'Interpreter','latex');
legend('Koopman','Linear','Location','northeast','FontSize',legend_fs,'Interpreter','latex');

out_pdf2 = fullfile(fig_dir, 'transient_metrics_summary_2.pdf');
out_png2 = fullfile(fig_dir, 'transient_metrics_summary_2.png');

exportgraphics(fig_sum2, out_pdf2, 'ContentType','vector');
exportgraphics(fig_sum2, out_png2, 'Resolution', 300);