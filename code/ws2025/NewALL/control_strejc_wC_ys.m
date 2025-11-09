%% Open ELab in MANAGER mode
elab_manager = ELab();
elab_manager.list();

%% Open ELab in CONTROL mode
Ts = 1;
device_name = 'pct23';
mode = 'control';
address = 'http://192.168.1.108:3030';
logging = 0;
logging_period = Ts;
internal_sampling_period = Ts;
polling_period = Ts;

pct23 = ELab(device_name, mode, address, logging, logging_period, internal_sampling_period, polling_period);

%% Device handshake
tags = pct23.getAllTags();
pct23.setTag('FSV',1);
pct23.setTag('Pump1',100);

%% Runtime configuration
sim_minutes = 12;
setpoint_T4 = 60;
Ts_runtime = Ts;
num_steps = sim_minutes * 60 / Ts_runtime;

Pump1_const = 65.8;
Heater_const = 25.0;

y_T4 = zeros(num_steps,1);
u_Pump2 = zeros(num_steps,1);
time_log = strings(num_steps,1);

pct23.setTag('FSV',1);
pct23.setTag('Pump1', Pump1_const);
pct23.setTag('Heater', Heater_const);

control_strejc(0, setpoint_T4, struct('reset', true));

%% Strejc MPC loop
for k = 1:num_steps
    tic;
    time_log(k) = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');

    T4 = double(pct23.getTag('T4').value);
    y_T4(k) = T4;

    u_cmd = control_strejc(T4, setpoint_T4);
    u_cmd = min(max(u_cmd, 0), 100);
    u_Pump2(k) = u_cmd;

    pct23.setTag('Pump2', u_cmd);
    pct23.setTag('Pump1', Pump1_const);
    pct23.setTag('Heater', Heater_const);
    pct23.setTag('FSV',1);

    elapsedTime = toc;
    pause(max(0, Ts_runtime - elapsedTime));
end

log_data = table((1:num_steps).', time_log, y_T4, u_Pump2, ...
    'VariableNames', {'step','timestamp','T4','Pump2'});
save('runtime_log_strejc.mat','log_data','setpoint_T4','Pump1_const','Heater_const');

pct23.off();
pct23.setTag('FSV',1);
control_strejc(0, setpoint_T4, struct('reset', true));

