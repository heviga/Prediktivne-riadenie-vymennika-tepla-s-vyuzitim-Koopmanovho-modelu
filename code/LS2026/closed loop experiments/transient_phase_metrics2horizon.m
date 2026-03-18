%% Auto-plot + metrics for ALL start temperatures
% Dynamic transient evaluation only
% All metrics refer only to the evaluated transient horizon Ntr
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

u_mean = 65.8447;
u_std  = 22.9062;

%% ===== SETTINGS =====
time_format = 'yyyy-MM-dd HH:mm:ss.SSS';

Q_cost = 10;
R_cost = 1;

temps = [45, 50, 55, 58, 60, 62, 66, 68];

% target used in plots / evaluation
x_mean_target = 59.0676 + abs(59.0676 - 63.8084);

%% ===== DYNAMIC TRANSIENT SETTINGS =====
settling_band = 1;              % +/- band around target
min_hold = 1;                   % window length
inside_fraction_required = 0.8; % at least 80 percent of samples in window inside band

%% ===== TABLE STORAGE =====
all_rows = [];

%% ===== VECTORS FOR METRIC PLOTS =====
T0_vec = temps(:);

TS_K   = nan(length(temps),1); TS_S   = nan(length(temps),1);
NTR_K  = nan(length(temps),1); NTR_S  = nan(length(temps),1);

IAEtrN_K  = nan(length(temps),1); IAEtrN_S  = nan(length(temps),1);
ISEtrN_K  = nan(length(temps),1); ISEtrN_S  = nan(length(temps),1);
ITAEtrN_K = nan(length(temps),1); ITAEtrN_S = nan(length(temps),1);
RMSEtr_K  = nan(length(temps),1); RMSEtr_S  = nan(length(temps),1);

PEAK_K  = nan(length(temps),1); PEAK_S  = nan(length(temps),1);
OVER_K  = nan(length(temps),1); OVER_S  = nan(length(temps),1);
UNDER_K = nan(length(temps),1); UNDER_S = nan(length(temps),1);

TVUn_K   = nan(length(temps),1); TVUn_S   = nan(length(temps),1);
DUMAX_K  = nan(length(temps),1); DUMAX_S  = nan(length(temps),1);
UENn_K   = nan(length(temps),1); UENn_S   = nan(length(temps),1);
OBJTRn_K = nan(length(temps),1); OBJTRn_S = nan(length(temps),1);

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

    %% ===== DYNAMIC TRANSIENT METRICS =====
    [ts_idx_koop, tr_koop] = compute_dynamic_transient_metrics( ...
        T4_koop, Pump_koop, target, x_std, u_mean, u_std, Q_cost, R_cost, ...
        settling_band, min_hold, inside_fraction_required);

    [ts_idx_strejc, tr_strejc] = compute_dynamic_transient_metrics( ...
        T4_strejc, Pump_strejc, target, x_std, u_mean, u_std, Q_cost, R_cost, ...
        settling_band, min_hold, inside_fraction_required);

    %% ===== STORE FOR SUMMARY PLOTS =====
    TS_K(k)  = tr_koop.SettlingTime;
    TS_S(k)  = tr_strejc.SettlingTime;

    NTR_K(k) = tr_koop.TransientLength;
    NTR_S(k) = tr_strejc.TransientLength;

    IAEtrN_K(k)  = tr_koop.IAE_Transient_Norm;
    IAEtrN_S(k)  = tr_strejc.IAE_Transient_Norm;
    ISEtrN_K(k)  = tr_koop.ISE_Transient_Norm;
    ISEtrN_S(k)  = tr_strejc.ISE_Transient_Norm;
    ITAEtrN_K(k) = tr_koop.ITAE_Transient_Norm;
    ITAEtrN_S(k) = tr_strejc.ITAE_Transient_Norm;
    RMSEtr_K(k)  = tr_koop.RMSE_Transient;
    RMSEtr_S(k)  = tr_strejc.RMSE_Transient;

    PEAK_K(k)  = tr_koop.PeakDeviation;
    PEAK_S(k)  = tr_strejc.PeakDeviation;
    OVER_K(k)  = tr_koop.Overshoot;
    OVER_S(k)  = tr_strejc.Overshoot;
    UNDER_K(k) = tr_koop.Undershoot;
    UNDER_S(k) = tr_strejc.Undershoot;

    TVUn_K(k)   = tr_koop.TVu_Transient_Norm;
    TVUn_S(k)   = tr_strejc.TVu_Transient_Norm;
    DUMAX_K(k)  = tr_koop.DeltaUMax_Transient;
    DUMAX_S(k)  = tr_strejc.DeltaUMax_Transient;
    UENn_K(k)   = tr_koop.ControlEnergy_Transient_Norm;
    UENn_S(k)   = tr_strejc.ControlEnergy_Transient_Norm;
    OBJTRn_K(k) = tr_koop.ObjectiveScaled_Transient_Norm;
    OBJTRn_S(k) = tr_strejc.ObjectiveScaled_Transient_Norm;

    %% ===== PRINT =====
    fprintf(['T0=%d | DYNAMIC TRANSIENT only: ' ...
             'Koop Ts=%s N=%d IAE/N=%.4f ISE/N=%.4f ITAE/N=%.4f RMSE=%.4f Peak=%.4f Over=%.4f Under=%.4f TVu/N=%.4f dUmax=%.4f Uen/N=%.4f Obj/N=%.4f | ' ...
             'Strejc Ts=%s N=%d IAE/N=%.4f ISE/N=%.4f ITAE/N=%.4f RMSE=%.4f Peak=%.4f Over=%.4f Under=%.4f TVu/N=%.4f dUmax=%.4f Uen/N=%.4f Obj/N=%.4f\n'], ...
        T0, ...
        num2str_or_nan(tr_koop.SettlingTime), tr_koop.TransientLength, ...
        tr_koop.IAE_Transient_Norm, tr_koop.ISE_Transient_Norm, tr_koop.ITAE_Transient_Norm, tr_koop.RMSE_Transient, ...
        tr_koop.PeakDeviation, tr_koop.Overshoot, tr_koop.Undershoot, ...
        tr_koop.TVu_Transient_Norm, tr_koop.DeltaUMax_Transient, tr_koop.ControlEnergy_Transient_Norm, tr_koop.ObjectiveScaled_Transient_Norm, ...
        num2str_or_nan(tr_strejc.SettlingTime), tr_strejc.TransientLength, ...
        tr_strejc.IAE_Transient_Norm, tr_strejc.ISE_Transient_Norm, tr_strejc.ITAE_Transient_Norm, tr_strejc.RMSE_Transient, ...
        tr_strejc.PeakDeviation, tr_strejc.Overshoot, tr_strejc.Undershoot, ...
        tr_strejc.TVu_Transient_Norm, tr_strejc.DeltaUMax_Transient, tr_strejc.ControlEnergy_Transient_Norm, tr_strejc.ObjectiveScaled_Transient_Norm);

    %% ===== STORE TABLE =====
    all_rows = [all_rows; ...
        table(T0, "Koopman", ...
              tr_koop.SettlingTime, tr_koop.SettlingIndex, tr_koop.SettlingFound, tr_koop.TransientLength, ...
              tr_koop.IAE_Transient_Norm, tr_koop.ISE_Transient_Norm, tr_koop.ITAE_Transient_Norm, tr_koop.RMSE_Transient, ...
              tr_koop.PeakDeviation, tr_koop.Overshoot, tr_koop.Undershoot, ...
              tr_koop.TVu_Transient_Norm, tr_koop.DeltaUMax_Transient, tr_koop.ControlEnergy_Transient_Norm, tr_koop.ObjectiveScaled_Transient_Norm, ...
              'VariableNames', {'T0','Controller', ...
              'SettlingTime','SettlingIndex','SettlingFound','TransientLength', ...
              'IAE_Transient_Norm','ISE_Transient_Norm','ITAE_Transient_Norm','RMSE_Transient', ...
              'PeakDeviation','Overshoot','Undershoot', ...
              'TVu_Transient_Norm','DeltaUMax_Transient','ControlEnergy_Transient_Norm','ObjectiveScaled_Transient_Norm'}); ...
        table(T0, "Strejc", ...
              tr_strejc.SettlingTime, tr_strejc.SettlingIndex, tr_strejc.SettlingFound, tr_strejc.TransientLength, ...
              tr_strejc.IAE_Transient_Norm, tr_strejc.ISE_Transient_Norm, tr_strejc.ITAE_Transient_Norm, tr_strejc.RMSE_Transient, ...
              tr_strejc.PeakDeviation, tr_strejc.Overshoot, tr_strejc.Undershoot, ...
              tr_strejc.TVu_Transient_Norm, tr_strejc.DeltaUMax_Transient, tr_strejc.ControlEnergy_Transient_Norm, tr_strejc.ObjectiveScaled_Transient_Norm, ...
              'VariableNames', {'T0','Controller', ...
              'SettlingTime','SettlingIndex','SettlingFound','TransientLength', ...
              'IAE_Transient_Norm','ISE_Transient_Norm','ITAE_Transient_Norm','RMSE_Transient', ...
              'PeakDeviation','Overshoot','Undershoot', ...
              'TVu_Transient_Norm','DeltaUMax_Transient','ControlEnergy_Transient_Norm','ObjectiveScaled_Transient_Norm'})];

    %% ===== PLOT ONLY EVALUATED TRANSIENT =====
    fig = figure('Color','w','Position',[100 100 900 520]);
    tiledlayout(2,1,'TileSpacing','Compact','Padding','Compact');

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

%% ===== SHOW / SAVE ONLY TRANSIENT TABLE =====
disp('===== PER-T0 DYNAMIC TRANSIENT METRICS =====');
disp(all_rows);

writetable(all_rows, fullfile(res_dir, 'per_T0_metrics_dynamic_only.csv'));

disp('Saved:');
disp(fullfile(res_dir, 'per_T0_metrics_dynamic_only.csv'));

%% ===== METRICS vs INITIAL CONDITION =====
plot_metric_figure(T0_vec, TS_K, TS_S, 'Settling time', 'Settling time (samples)', ...
    fullfile(fig_dir, 'metric_SettlingTime.png'));

plot_metric_figure(T0_vec, NTR_K, NTR_S, 'Transient length', 'Transient length (samples)', ...
    fullfile(fig_dir, 'metric_TransientLength.png'));

plot_metric_figure(T0_vec, IAEtrN_K, IAEtrN_S, 'IAE transient / N', 'IAE / N', ...
    fullfile(fig_dir, 'metric_IAEtr_Norm.png'));

plot_metric_figure(T0_vec, ISEtrN_K, ISEtrN_S, 'ISE transient / N', 'ISE / N', ...
    fullfile(fig_dir, 'metric_ISEtr_Norm.png'));

plot_metric_figure(T0_vec, ITAEtrN_K, ITAEtrN_S, 'ITAE transient / N', 'ITAE / N', ...
    fullfile(fig_dir, 'metric_ITAEtr_Norm.png'));

plot_metric_figure(T0_vec, RMSEtr_K, RMSEtr_S, 'RMSE transient', 'RMSE transient', ...
    fullfile(fig_dir, 'metric_RMSEtr.png'));

plot_metric_figure(T0_vec, PEAK_K, PEAK_S, 'Peak deviation', 'Peak deviation ($^\circ$C)', ...
    fullfile(fig_dir, 'metric_Peak.png'));

plot_metric_figure(T0_vec, OVER_K, OVER_S, 'Overshoot', 'Overshoot ($^\circ$C)', ...
    fullfile(fig_dir, 'metric_Overshoot.png'));

plot_metric_figure(T0_vec, UNDER_K, UNDER_S, 'Undershoot', 'Undershoot ($^\circ$C)', ...
    fullfile(fig_dir, 'metric_Undershoot.png'));

plot_metric_figure(T0_vec, TVUn_K, TVUn_S, 'TVu transient / N', 'TVu / N', ...
    fullfile(fig_dir, 'metric_TVu_Norm.png'));

plot_metric_figure(T0_vec, DUMAX_K, DUMAX_S, 'Maximum input increment', 'Max $|\Delta u|$', ...
    fullfile(fig_dir, 'metric_DeltaUMax.png'));

plot_metric_figure(T0_vec, UENn_K, UENn_S, 'Control energy / N', 'Control energy / N', ...
    fullfile(fig_dir, 'metric_ControlEnergy_Norm.png'));

plot_metric_figure(T0_vec, OBJTRn_K, OBJTRn_S, 'Transient scaled objective / N', 'Objective / N', ...
    fullfile(fig_dir, 'metric_ObjectiveTransient_Norm.png'));

%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================

function [ts_idx, metrics] = compute_dynamic_transient_metrics( ...
    y, u, target, x_std, u_mean, u_std, Q_cost, R_cost, ...
    settling_band, min_hold, inside_fraction_required)

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

    Ntr = length(idx_tr);
    metrics.TransientLength = Ntr;

    % raw transient metrics
    iae_tr  = sum(abs(e_tr));
    ise_tr  = sum(e_tr.^2);
    itae_tr = sum((0:Ntr-1)' .* abs(e_tr));
    rmse_tr = sqrt(mean(e_tr.^2));

    [~, idx_peak] = max(abs(e_tr));
    peak_tr = abs(e_tr(idx_peak));

    over_tr  = max(0, max(y_tr - target));
    under_tr = max(0, max(target - y_tr));

    if length(u_tr) > 1
        du = diff(u_tr);
        tvu_tr   = sum(abs(du));
        dumax_tr = max(abs(du));
    else
        tvu_tr   = 0;
        dumax_tr = 0;
    end

    uen_tr = sum(u_tr.^2);

    objtr_scaled = sum( ...
        Q_cost*((y_tr - target)/x_std).^2 + ...
        R_cost*((u_tr - u_mean)/u_std).^2 );

    % normalized metrics on transient horizon
    metrics.IAE_Transient_Norm  = iae_tr  / Ntr;
    metrics.ISE_Transient_Norm  = ise_tr  / Ntr;
    metrics.ITAE_Transient_Norm = itae_tr / Ntr;
    metrics.RMSE_Transient      = rmse_tr;
    metrics.PeakDeviation       = peak_tr;
    metrics.PeakIndex           = idx_peak;
    metrics.Overshoot           = over_tr;
    metrics.Undershoot          = under_tr;
    metrics.TVu_Transient_Norm  = tvu_tr / Ntr;
    metrics.DeltaUMax_Transient = dumax_tr;
    metrics.ControlEnergy_Transient_Norm = uen_tr / Ntr;
    metrics.ObjectiveScaled_Transient_Norm = objtr_scaled / Ntr;
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