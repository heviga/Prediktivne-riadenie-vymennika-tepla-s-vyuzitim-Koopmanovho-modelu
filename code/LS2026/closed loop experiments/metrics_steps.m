%% Auto-plot + metrics ONLY for start temperature >= 60 °C
clc; clear;close all;

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
x_mean_target = 59.0676 + abs(59.0676 - 63.8084);

%% ===== GLOBAL ACCUMULATORS =====
sum_obj_koop_scaled_all   = 0;
sum_obj_strejc_scaled_all = 0;

sum_iae_koop_all   = 0;
sum_iae_strejc_all = 0;

sum_rmse_koop_all   = 0;
sum_rmse_strejc_all = 0;


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
    time_koop = minutes(t - t(1));

    T4_koop   = log_data.T4(:);
    Pump_koop = log_data.Pump2(:);

    %% ===== LOAD STREJC =====
    S = load(strejc_file);
    log_data = S.log_data;

    t = datetime(string(log_data.timestamp), 'InputFormat', time_format);
    time_strejc = minutes(t - t(1));

    T4_strejc   = log_data.T4(:);
    Pump_strejc = log_data.Pump2(:);

    %% ===== METRICS =====
    target = x_mean_target;
    bias   = target - x_mean;

    % --- RMSE ---
    rmse_koop   = sqrt(mean((T4_koop   - target).^2));
    rmse_strejc = sqrt(mean((T4_strejc - target).^2));
    sum_rmse_koop_all   = sum_rmse_koop_all   + rmse_koop;
    sum_rmse_strejc_all = sum_rmse_strejc_all + rmse_strejc;


    % --- IAE ---
    iae_koop   = sum(abs(T4_koop   - target));
    iae_strejc = sum(abs(T4_strejc - target));

    % --- Objective (physical) ---
    obj_koop = sum(Q_cost*(T4_koop - target).^2 ...
                 + R_cost*(Pump_koop).^2);

    obj_strejc = sum(Q_cost*(T4_strejc - target).^2 ...
                   + R_cost*(Pump_strejc).^2);

    % --- Scaled objective ---
    obj_koop_scaled = sum( ...
        Q_cost*((T4_koop   - bias - x_mean)/x_std).^2 + ...
        R_cost*((Pump_koop - u_mean)/u_std).^2 );

    obj_strejc_scaled = sum( ...
        Q_cost*((T4_strejc - bias - x_mean)/x_std).^2 + ...
        R_cost*((Pump_strejc - u_mean)/u_std).^2 );

    % --- Accumulate global sums ---
    sum_obj_koop_scaled_all   = sum_obj_koop_scaled_all   + obj_koop_scaled;
    sum_obj_strejc_scaled_all = sum_obj_strejc_scaled_all + obj_strejc_scaled;

    sum_iae_koop_all   = sum_iae_koop_all   + iae_koop;
    sum_iae_strejc_all = sum_iae_strejc_all + iae_strejc;

    %% ===== PER-EXPERIMENT TABLE =====
    metrics = table( ...
        {'Koopman'; 'Strejc'}, ...
        [rmse_koop; rmse_strejc], ...
        [iae_koop; iae_strejc], ...
        [obj_koop; obj_strejc], ...
        [obj_koop_scaled; obj_strejc_scaled], ...
        'VariableNames', ...
        {'Controller','RMSE_T4','IAE','Objective','Objective_Scaled'} );

    disp(metrics);

end

metrics_total = table( ...
    {'Koopman'; 'Strejc'}, ...
    [sum_rmse_koop_all; sum_rmse_strejc_all], ...
    [sum_iae_koop_all;  sum_iae_strejc_all], ...
    [sum_obj_koop_scaled_all; sum_obj_strejc_scaled_all], ...
    'VariableNames', ...
    {'Controller','Total_RMSE','Total_IAE','Total_Objective_Scaled'} );

disp('===== FINAL SUM OVER ALL INITIAL CONDITIONS =====');
disp(metrics_total);

