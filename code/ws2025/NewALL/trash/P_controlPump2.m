%% === CONTROL: Pump2 + Heater to reach setpoint T4 (PI) ===

% Device / sampling setup
Ts = 1;
device_name = 'pct23';
mode = 'control';
address = 'http://192.168.1.108:3030';
logging = 0;
logging_period = Ts;
internal_sampling_period = Ts;
polling_period = Ts;

pct23 = ELab(device_name, mode, address, logging, logging_period, internal_sampling_period, polling_period);

%% Test / initialize
pct23.setTag('FSV',1);
pct23.setTag('Pump1',100);

%% Runtime configuration (men setpoint)
sim_minutes = 7;
setpoint_T4 = 50;    % desired temperature setpoint
Ts_runtime = Ts;
num_steps = sim_minutes * 60 / Ts_runtime;
P_spiral = 12;

% Pump / Heater constants
Pump1_const = 50;
Pump2_base = 65;  % základná hodnota Pump2
Pump2_min = 0;
Pump2_max = 100;
Kp_pump2 = 15.0;   % P-gain pre Pump2 reguláciu
Ki_pump2 = 2;   % I-gain pre Pump2 reguláciu

% Initialize integral error
integral_error = 0;

% Preallocate logs
y_T4 = zeros(num_steps,1);
y_T2 = zeros(num_steps,1);
u_Heater = zeros(num_steps,1);
u_Pump2 = zeros(num_steps,1);
time_log = strings(num_steps,1);

% Reset Strejc / kontroler ak používaš
control_strejc(0, struct('reset', true));

% Quick read to ensure tags are available
double(pct23.getTag('T4').value); pause(1);
double(pct23.getTag('T2').value); pause(1);
double(pct23.getTag('T1').value);

%% ====== Main control loop ======
for k = 1:num_steps
    tic;
    time_log(k) = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');

    % --- Measurements ---
    T4 = double(pct23.getTag('T4').value);
    T2 = double(pct23.getTag('T2').value);
    y_T4(k) = T4;
    y_T2(k) = T2;

    % --- Heater command (simple P regulator for spiral) ---
    value_sp = min(max(P_spiral * (76 - T2), 0), 100);
    u_Heater(k) = value_sp;

    % --- Pump2 command (PI controller to reach T4 setpoint) ---
    error = setpoint_T4 - T4;
    integral_error = integral_error + error*Ts_runtime;

    u_Pump2(k) = Pump2_base + Kp_pump2*error + Ki_pump2*integral_error;
    u_Pump2(k) = min(max(u_Pump2(k), Pump2_min), Pump2_max);  % saturácia

    % --- Apply control to device ---
    pct23.setTag('Pump2', u_Pump2(k));
    pct23.setTag('Pump1', Pump1_const);
    pct23.setTag('Heater', u_Heater(k));
    pct23.setTag('FSV', 1);

    % --- Display info ---
    fprintf('loop: %d, T4 = %.3f °C, Pump2 = %.3f %%, Heater = %.3f %%\n', k, T4, u_Pump2(k), value_sp);

    % --- Maintain sampling time ---
    pause(max(0, Ts_runtime - toc));
end

%% ====== Logging and cleanup ======
xmean_T4 = mean(y_T4);
xmean_T2 = mean(y_T2);

fprintf('\n=== XMEAN RESULTS ===\n');
fprintf('xmean T4 = %.4f °C\n', xmean_T4);
fprintf('xmean T2 = %.4f °C\n', xmean_T2);

% Save log
% log_data = table((1:num_steps).', time_log, y_T4, y_T2, u_Pump2, u_Heater, ...
%     'VariableNames', {'step','timestamp','T4','T2','Pump2','Heater'});
% save('runtime_log_T4_control_PI.mat','log_data','setpoint_T4','Pump1_const');

% Turn device off / cleanup
pct23.off();
pct23.setTag('FSV',1);
control_strejc(0, struct('reset', true));

disp('Pump2 + Heater PI control runtime finished.');