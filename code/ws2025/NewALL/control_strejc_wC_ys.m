%% Open ELab in MANAGER mode
elab_manager = ELab();
elab_manager.list();

%% Open ELab in CONTROL mode
Ts = 1; % sampling period
device_name = 'pct23';
mode = 'control';
address = 'http://192.168.1.108:3030';
logging = 0;
logging_period = Ts;
internal_sampling_period = Ts;
polling_period = Ts;

pct23 = ELab(device_name, mode, address, logging, logging_period, internal_sampling_period, polling_period);

%% Test device
pct23.setTag('FSV',1);
pct23.setTag('Pump1',100)

%% Runtime configuration
sim_minutes = 5;
%setpoint_T4 = 58.3377; % desired temperature
Ts_runtime = Ts;
num_steps = sim_minutes * 60 / Ts_runtime;
P_spiral = 12;
Pump1_const = 50;

% Preallocate logs
y_T4 = zeros(num_steps,1);
y_T2 = zeros(num_steps,1);
u_Heater = zeros(num_steps,1);
u_Pump2 = zeros(num_steps,1);
time_log = strings(num_steps,1);

ys = [59.0676 - 59.824];
Pump1_const = 50;
% Reset Strejc controller
control_strejc(0, struct('reset', true));

double(pct23.getTag('T4').value)
pause(1) 
double(pct23.getTag('T2').value)
pause(1)
double(pct23.getTag('T1').value)


%% Main control loop
for k = 1:num_steps
    tic;
    time_log(k) = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');

    % --- Measurements ---
    T4 = double(pct23.getTag('T4').value);
    T2 = double(pct23.getTag('T2').value);

    y_T4(k) = T4;
    y_T2(k) = T2;

     % --- Compute Heater command (simple P regulator) ---
    value_sp = P_spiral * (76 - T2); % P regulator for spiral
    value_sp = min(max(value_sp, 0), 100);

    fprintf('T4 = %.3f\n', y_T4(k));

    % --- Strejc MPC control ---
    u_cmd = control_strejc(T4 + ys); % t4 + ys = offset od odchylky od identifikacie - minimalny
    u_cmd = min(max(u_cmd,0),100);
    u_Pump2(k) = u_cmd;

    % --- Apply control ---
    pct23.setTag('Pump2', u_Pump2(k));
    fprintf('loop: %d, Pump2 = %.3f\n', k, u_Pump2(k));
    pct23.setTag('Pump1', Pump1_const);
    pct23.setTag('Heater', value_sp);
    pct23.setTag('FSV', 1);

    % --- Maintain sampling time ---
    elapsedTime = toc;
    pause(max(0, Ts_runtime - elapsedTime));
end

%% ===============================
%   LOGGING AND CLEANUP
%   ===============================

log_data = table((1:num_steps).', time_log, y_T4, u_Pump2, ...
    'VariableNames', {'step','timestamp','T4','Pump2'});

save('runtime_log_strejc.mat','log_data','setpoint_T4','Pump1_const');

pct23.off();
pct23.setTag('FSV',1);
control_strejc(0, struct('reset', true));

disp('Strejc control runtime finished.');
