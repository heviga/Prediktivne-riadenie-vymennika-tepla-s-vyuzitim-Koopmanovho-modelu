%% Plot Koopman vs Strejc logs (T4 & Pump2)
% Each MAT file must contain table `log_data` with timestamp, T4, Pump2.

koop_file = 'runtime_log_koop.mat';
strejc_file = 'runtime_log_strejc.mat';
time_format = 'yyyy-MM-dd HH:mm:ss.SSS';
required_vars = {'timestamp','T4','Pump2'};

% --- Koopman log ---
S = load(koop_file);
log_data = S.log_data;
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
S = load(strejc_file);
log_data = S.log_data;
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
legend({'Koopman','Strejc'}, 'Location','southeast');

nexttile;
plot(time_koop, Pump_koop, 'LineWidth', 1.3);
hold on;
plot(time_strejc, Pump_strejc, 'LineWidth', 1.3);
grid on;
ylabel('Pump2 [%]');
xlabel('Elapsed time [min]');
title('Pump2 Actuation');
legend({'Koopman','Strejc'}, 'Location','southeast');

sgtitle('Koopman vs Strejc control logs');
