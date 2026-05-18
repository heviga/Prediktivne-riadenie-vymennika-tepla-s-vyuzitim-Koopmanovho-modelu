%% Plot Koopman vs Strejc logs (T4 & Pump2)
% Each MAT file must contain table `log_data` with timestamp, T4, Pump2.

koop_file = 'steps/runtime_log_koop10.mat';
strejc_file = 'steps/runtime_log_strejc10.mat';
time_format = 'yyyy-MM-dd HH:mm:ss.SSS';
required_vars = {'timestamp','T4','Pump2'};
Q_cost = 10;
R_cost = 1;
x_mean_target = 59.0676 + abs(59.0676 - 63.8084) % target temperature

% --- Koopman log ---
S_koop = load(koop_file);
log_data = S_koop.log_data;
missing = setdiff(required_vars, log_data.Properties.VariableNames);
assert(isempty(missing), 'Koopman log missing vars: %s', strjoin(missing, ', '));

time_raw = log_data.timestamp;
if iscell(time_raw)
    time_raw = string(time_raw);
end
time_vals = datetime(time_raw, 'InputFormat', time_format);
time_koop = minutes(time_vals - time_vals(1));
T4_koop = log_data.T4;
Pump_koop = log_data.Pump2;

% --- Strejc log ---
S_strejc = load(strejc_file);
log_data = S_strejc.log_data;
missing = setdiff(required_vars, log_data.Properties.VariableNames);
assert(isempty(missing), 'Strejc log missing vars: %s', strjoin(missing, ', '));

time_raw = log_data.timestamp;
if iscell(time_raw)
    time_raw = string(time_raw);
end
time_vals = datetime(time_raw, 'InputFormat', time_format);
time_strejc = minutes(time_vals - time_vals(1));
T4_strejc = log_data.T4;
Pump_strejc = log_data.Pump2;

figure('Name','Comparison: T4 & Pump2','Color','w');
tiledlayout(2,1,'TileSpacing','Compact','Padding','Compact');

nexttile;
plot(time_koop, T4_koop, 'LineWidth', 1.3);
hold on;
plot(time_strejc, T4_strejc, 'LineWidth', 1.3);
grid on;
ylabel('T4 [°C]');
title('Measured Temperature');
legend({'Koopman','Linear'}, 'Location','southeast');

nexttile;
plot(time_koop, Pump_koop, 'LineWidth', 1.3);
hold on;
plot(time_strejc, Pump_strejc, 'LineWidth', 1.3);
grid on;
ylabel('Pump2 [%]');
xlabel('Elapsed time [min]');
title('Pump2 Actuation');
legend({'Koopman','Linear'}, 'Location','southeast');

sgtitle('Koopman vs linear control logs');

%% Metrics
target_koop = x_mean_target;
target_strejc = x_mean_target;

rmse_koop = sqrt(mean((T4_koop - target_koop).^2));
rmse_strejc = sqrt(mean((T4_strejc - target_strejc).^2));

obj_koop = sum(Q_cost*(T4_koop - target_koop).^2 + R_cost*(Pump_koop).^2);
obj_strejc = sum(Q_cost*(T4_strejc - target_strejc).^2 + R_cost*(Pump_strejc).^2);

sum_pump_koop = sum(Pump_koop);
sum_pump_strejc = sum(Pump_strejc);

metrics = table( ...
    {'Koopman'; 'Strejc'}, ...
    [rmse_koop; rmse_strejc], ...
    [obj_koop; obj_strejc], ...
    [sum_pump_koop; sum_pump_strejc], ...
    'VariableNames', {'Controller','RMSE_T4','Objective','SumPump2'});

disp('--- Performance metrics ---');
disp(metrics);

%% Save metrics
metrics_file_mat = 'steps/metrics_comparison10.mat';
metrics_file_csv = 'steps/metrics_comparison10.csv';

save(metrics_file_mat, 'metrics');
writetable(metrics, metrics_file_csv);

disp(['Metrics saved to: ', metrics_file_mat]);
disp(['Metrics saved to: ', metrics_file_csv]);
