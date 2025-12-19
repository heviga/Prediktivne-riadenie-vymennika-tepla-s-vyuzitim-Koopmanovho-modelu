%% Auto-plot + metrics for all Koopman/Strejc logs

time_format = 'yyyy-MM-dd HH:mm:ss.SSS';
required_vars = {'timestamp','T4','Pump2'};

Q_cost = 10;
R_cost = 1;

temps = [45, 50, 55, 58, 60, 62, 66, 68];   % initial temperatures
x_mean_target = 59.0676 + abs(59.0676 - 63.8084);

for k = 1:length(temps)

    T0 = temps(k);  % initial temperature to label files & figures

    koop_file = sprintf('steps/2611runtime_log_koop%d.mat', k);
    strejc_file = sprintf('steps/2611runtime_log_strejc%d.mat', k);

    fprintf('\n--- Processing start T = %d °C ---\n', T0);

    %% ===== LOAD KOOPMAN =====
    S_koop = load(koop_file);
    log_data = S_koop.log_data;

    time_raw = log_data.timestamp;
    if iscell(time_raw), time_raw = string(time_raw); end
    t = datetime(time_raw, 'InputFormat', time_format);
    time_koop = minutes(t - t(1));

    T4_koop = log_data.T4;
    Pump_koop = log_data.Pump2;

    %% ===== LOAD STREJC =====
    S_strejc = load(strejc_file);
    log_data = S_strejc.log_data;

    time_raw = log_data.timestamp;
    if iscell(time_raw), time_raw = string(time_raw); end
    t = datetime(time_raw, 'InputFormat', time_format);
    time_strejc = minutes(t - t(1));

    T4_strejc = log_data.T4;
    Pump_strejc = log_data.Pump2;


    %% ===== FIGURE with 2 tiles =====
    fig = figure('Name', sprintf('Start %d°C — Koopman vs Strejc', T0), 'Color', 'w');
    tiledlayout(2,1,'TileSpacing','Compact','Padding','Compact');

    % --- T4 ---
    nexttile;
    plot(time_koop, T4_koop, 'LineWidth', 1.3); hold on;
    plot(time_strejc, T4_strejc, 'LineWidth', 1.3);
    grid on; ylabel('T4 [°C]');
    title(sprintf('Measured Temperature (Start %d°C)', T0));
    legend({'Koopman','Strejc'}, 'Location','southeast');

    % --- Pump2 ---
    nexttile;
    plot(time_koop, Pump_koop, 'LineWidth', 1.3); hold on;
    plot(time_strejc, Pump_strejc, 'LineWidth', 1.3);
    grid on; ylabel('Pump2 [%]'); xlabel('Elapsed time [min]');
    title('Pump2 Actuation');
    legend({'Koopman','Strejc'}, 'Location','southeast');

    sgtitle(sprintf('Koopman vs Strejc Control Logs — Start %d°C', T0));


    %% ===== METRICS =====
    target = x_mean_target;

    rmse_koop = sqrt(mean((T4_koop - target).^2));
    rmse_strejc = sqrt(mean((T4_strejc - target).^2));

    obj_koop = sum(Q_cost*(T4_koop - target).^2 + R_cost*(Pump_koop).^2);
    obj_strejc = sum(Q_cost*(T4_strejc - target).^2 + R_cost*(Pump_strejc).^2);

    sum_pump_koop = sum(Pump_koop);
    sum_pump_strejc = sum(Pump_strejc);

    metrics = table( ...
        {'Koopman'; 'Strejc'}, ...
        [rmse_koop; rmse_strejc], ...
        [obj_koop; obj_strejc], ...
        [sum_pump_koop; sum_pump_strejc], ...
        'VariableNames', {'Controller','RMSE_T4','Objective','SumPump2'});

    disp('--- Performance metrics ---');
    fprintf('starting temperature: %d\n', T0);
    disp(metrics);


    %% ===== SAVE METRICS =====
%     metrics_mat = sprintf('steps/metrics_start%d.mat', T0);
%     metrics_csv = sprintf('steps/metrics_start%d.csv', T0);
% 
%     save(metrics_mat, 'metrics');
%     writetable(metrics, metrics_csv);
% 
%     fprintf('Metrics saved to %s and %s\n', metrics_mat, metrics_csv);

end

disp('--- All figures + metrics generated ---');
