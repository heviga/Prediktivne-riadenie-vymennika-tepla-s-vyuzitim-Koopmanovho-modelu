%% Open ELab in MANAGER mode
% volannie u_cmd = control_koopman2(T4, setpoint_T4);
                              % the device 'pct23'.
%skoky trenovacie: 30,50,70,90,70,50,30
%skoky testovacie: u = 60,30,80,100

elab_manager = ELab();
elab_manager.list();
% s u_mean namerat x -> x_REALmean, z x real offsety
%% Open ELab in CONTROL mode
%  In this mode, you have full control over selected device
%
%  Example: 
%           elab_manager = ELab(DEVICE_NAME, MODE, ADDRESS, LOGGING, LOGGING_PERIOD, INTERNAL_SAMPLING_PERIOD, POLLING_PERIOD); 
%
%           where DEVICE_NAME (String) is a designated name of the device (e.g. 'pct23'),
%                 MODE (String) is mode switch with possible values 'MANAGER', 'CONTROL', 'MONITOR',
%                 ADDRESS (String) is HTTP address of elab master SCADA system,
%                 LOGGING (0 or 1) is switch for online data logging into elab master database,
%                 LOGGING_PERIOD (N seconds) defines how often the measured data is logged into database,
%                 INTERNAL_SAMPLING_PERIOD (N seconds) defines how often the device streams new data to the elab SCADA master,
%                 POLLING_PERIOD (N seconds) defines how often the ELab class refreshes the data from SCADA master (this should be set to Ts)
%

Ts = 1;%
device_name = 'pct23';
mode = 'control';
address = 'http://192.168.1.108:3030';%
logging = 0;%
logging_period = Ts;
internal_sampling_period = Ts;
polling_period = Ts;

% create instance of udaq28 device
pct23 = ELab(device_name, mode, address, logging, logging_period, internal_sampling_period, polling_period);

%% Using the device (measure/control)

% get all measured data at once
tags = pct23.getAllTags();
%% test if working

pct23.setTag('FSV',1);
pct23.setTag('Pump1',100)

%% Runtime configuration
sim_minutes = 12;           % total runtime in minutes
%setpoint_T4 =58.3377;           % desired temperature setpoint (°C)
Ts_runtime = Ts;            % reuse device sampling period
num_steps = sim_minutes * 60 / Ts_runtime; %celkovy cas
P_spiral = 12;

%x_mean =  59.0676 + offset 
ys = [59.0676 - 59.824]; %nam];    % offset for T4 and T2 xmean - xreal
%us = [65.8447];           % steady-state inputs [Pump2, Heater]
% Auxiliary actuators held at steady state
Pump1_const = 50;
%Heater_const = 25.0;

% Preallocate logs
y_T4 = zeros(num_steps,1);
y_T2 = zeros(num_steps,1);
u_Pump2 = zeros(num_steps,1);
u_Heater = zeros(num_steps,1);
time_log = strings(num_steps,1);

% Reset Koopman controller
control_koopman(0, struct('reset', true));

% loading current measurements
double(pct23.getTag('T4').value)
pause(1) 
double(pct23.getTag('T2').value)
pause(1)
double(pct23.getTag('T1').value)



% Reset Koopman controller internal state
%control_koopman(0, setpoint_T4, struct('reset', true));



% for k = 1:num_steps
%     tic;
%     time_log(k) = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');
% 
%     % --- Measurements ---
%     T4 = double(pct23.getTag('T4').value);
%     T2 = double(pct23.getTag('T2').value);
%     y_T4(k) = T4;
%     y_T2(k) = T2;
% 
%     % --- Compute Heater command (simple P regulator) ---
%     value_sp = P_spiral * (76 - T2); % P regulator for spiral
%     value_sp = min(max(value_sp, 0), 100);
% 
%     % --- Koopman MPC control for Pump2 ---
%     % Add offset correction relative to steady-state
%     y_offset =(T4 + ys);
%     u_cmd = control_koopman(y_offset)%% Open ELab in MANAGER mode
% volannie u_cmd = control_koopman2(T4, setpoint_T4);
                              % the device 'pct23'.
%skoky trenovacie: 30,50,70,90,70,50,30
%skoky testovacie: u = 60,30,80,100


% s u_mean namerat x -> x_REALmean, z x real offsety
%% Open ELab in CONTROL mode
%  In this mode, you have full control over selected device
%
%  Example: 
%           elab_manager = ELab(DEVICE_NAME, MODE, ADDRESS, LOGGING, LOGGING_PERIOD, INTERNAL_SAMPLING_PERIOD, POLLING_PERIOD); 
%
%           where DEVICE_NAME (String) is a designated name of the device (e.g. 'pct23'),
%                 MODE (String) is mode switch with possible values 'MANAGER', 'CONTROL', 'MONITOR',
%                 ADDRESS (String) is HTTP address of elab master SCADA system,
%                 LOGGING (0 or 1) is switch for online data logging into elab master database,
%                 LOGGING_PERIOD (N seconds) defines how often the measured data is logged into database,
%                 INTERNAL_SAMPLING_PERIOD (N seconds) defines how often the device streams new data to the elab SCADA master,
%                 POLLING_PERIOD (N seconds) defines how often the ELab class refreshes the data from SCADA master (this should be set to Ts)
%

Ts = 1;%
device_name = 'pct23';
mode = 'control';
address = 'http://192.168.1.108:3030';%
logging = 0;%
logging_period = Ts;
internal_sampling_period = Ts;
polling_period = Ts;

% create instance of udaq28 device
pct23 = ELab(device_name, mode, address, logging, logging_period, internal_sampling_period, polling_period);

%% Using the device (measure/control)

% get all measured data at once
tags = pct23.getAllTags();
%% test if working

pct23.setTag('FSV',1);
pct23.setTag('Pump1',100)

%% Runtime configuration
sim_minutes = 5;           % total runtime in minutes
setpoint_T4 =58.3377;           % desired temperature setpoint (°C)
Ts_runtime = Ts;            % reuse device sampling period
num_steps = sim_minutes * 60 / Ts_runtime; %celkovy cas
P_spiral = 12;

%x_mean =  59.0676 + offset
ys = [59.0676 - 59.0676]; %nam];    % offsets for T4 and T2 xmean - xreal
%us = [65.8447];           % steady-state inputs [Pump2, Heater]
% Auxiliary actuators held at steady state
Pump1_const = 50;
%Heater_const = 25.0;

% Preallocate logs
y_T4 = zeros(num_steps,1);
y_T2 = zeros(num_steps,1);
u_Pump2 = zeros(num_steps,1);
u_Heater = zeros(num_steps,1);
time_log = strings(num_steps,1);

% Reset Koopman controller
control_koopman(0, struct('reset', true));

% loading current measurements
double(pct23.getTag('T4').value)
pause(1) 
double(pct23.getTag('T2').value)
pause(1)
double(pct23.getTag('T1').value)



% Reset Koopman controller internal state
%control_koopman(0, setpoint_T4, struct('reset', true));



for k = 1:num_steps
    tic;
    time_log(k) = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');

    % --- Measurements ---
    T4 = double(pct23.getTag('T4').value);
    T2 = double(pct23.getTag('T2').value);
    y_T4(k) = T4;
    y_T2(k) = T2;
    fprintf('T4 = %.3f\n', y_T4(k))


    % --- Compute Heater command (simple P regulator) ---
    value_sp = P_spiral * (76 - T2); % P regulator for spiral
    value_sp = min(max(value_sp, 0), 100);

    % --- Koopman MPC control for Pump2 ---
    % Add offset correction relative to steady-state
    y_offset =(T4 + ys)
    u_cmd = control_koopman(y_offset);
    u_cmd = min(max(u_cmd, 0), 100);
    u_Pump2(k) = u_cmd(1);

    % --- Apply control to device ---
    pct23.setTag('Pump2', u_Pump2(k));
    fprintf('loop: %d\n', k);
    fprintf('akcny zasah = %.3f\n', u_Pump2(k))
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

log_data = table((1:num_steps).', time_log, y_T4, y_T2, u_Pump2, u_Heater, ...
    'VariableNames', {'step','timestamp','T4','T2','Pump2','Heater'});

save('runtime_log_koop.mat','log_data','setpoint_T4','Pump1_const','ys');

pct23.off();
pct23.setTag('FSV',1);
control_koopman(0, struct('reset', true));



%zmenit nazov pri dalsich skokoch
%
