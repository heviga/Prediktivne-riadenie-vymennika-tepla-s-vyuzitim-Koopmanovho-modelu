%% Open ELab in MANAGER mode
%  In this mode, you can LIST and INSTALL devices
%
%  Example: 
%           elab_manager = ELab();  % Creating elab instance by calling 
%                                   % ELab class without parameters, 
%                                   % automatically triggers the MANAGER mode.
%           elab_manager.list();    % Displays list of devices available in
%                                   % elab master database.
%           
%           elab_manager.install('pct23'); % Installs library files for
%                                           % the device 'pct23'.
%

elab_manager = ELab();
elab_manager.list();

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
setTag('Pump1',100)

%% Runtime configuration
sim_minutes = 12;           % total runtime in minutes
setpoint_T4 = 60;           % desired temperature setpoint (°C)
Ts_runtime = Ts;            % reuse device sampling period
num_steps = sim_minutes * 60 / Ts_runtime; %celkovy cas

% Auxiliary actuators held at steady state
Pump1_const = 65.8;
Heater_const = 25.0;

% Preallocate logs
y_T4 = zeros(num_steps,1);
u_Pump2 = zeros(num_steps,1);
time_log = strings(num_steps,1);

% Ensure device is in a known state
pct23.setTag('FSV',1);
pct23.setTag('Pump1', Pump1_const);
pct23.setTag('Heater', Heater_const);

% Reset Koopman controller internal state
control_koopman(0, setpoint_T4, struct('reset', true));

%% Koopman MPC loop (single-output T4 control via Pump2)
for k = 1:num_steps
    tic;
    time_log(k) = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');

    % Measurement
    T4 = double(pct23.getTag('T4').value);
    y_T4(k) = T4;

    % Compute Pump2 command
    u_cmd = control_koopman(T4, setpoint_T4);
    u_cmd = min(max(u_cmd, 0), 100); % safety clip
    u_Pump2(k) = u_cmd;

    % Apply control
    pct23.setTag('Pump2', u_cmd);
    pct23.setTag('Pump1', Pump1_const);
    pct23.setTag('Heater', Heater_const);
    pct23.setTag('FSV',1);

    % Maintain sampling period
    elapsedTime = toc;
    pause(max(0, Ts_runtime - elapsedTime));
end

%zmenit nazov pri dalsich skokoch
% Persist logs for offline analysis
log_data = table((1:num_steps).', time_log, y_T4, u_Pump2, ...
    'VariableNames', {'step','timestamp','T4','Pump2'});
save('runtime_log.mat','log_data','setpoint_T4','Pump1_const','Heater_const');

%terminate(pyenv);
pct23.off();
pct23.setTag('FSV',1);
control_koopman(0, setpoint_T4, struct('reset', true));
