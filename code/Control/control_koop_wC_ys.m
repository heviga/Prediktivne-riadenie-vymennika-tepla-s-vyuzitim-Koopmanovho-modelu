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

%% simulation variables
d = 60;
N = 12*d;

y = zeros(N,3);
u = zeros(N,3);
u_prev = [50,50,20];

ys = [42.02774966-53.5, 66.97571158-72.5, 42.5152149-54] 
us = [65.81566067 54.53911806 25.00000754];

% loading current measurements
double(pct23.getTag('T4').value)
pause(1) 
double(pct23.getTag('T2').value)
pause(1)
double(pct23.getTag('T1').value)
% P as P regulator gain
P_spiral = 12;

%dame 50 percentne u (z umean) 
%% doprdele - steer your system to undesired location
for i = 1:5*d
    % Start the timer
    tic;

    % Code to execute within the loop
    disp(['Iteration: ', num2str(i), ' at ', datestr(now)]);

    y3 = double(pct23.getTag('T4').value);
    y2 = double(pct23.getTag('T2').value);
    y1 = double(pct23.getTag('T1').value);

    y(i,:) = [y1 y2 y3];
    y(i,:)
    pct23.setTag('FSV',1);%necceessary?

    value_sp =  P_spiral*(75 - y2); % simple P regulator teploty a simulink schema rozdiel
    value_pump =  P_spiral*(60 - y3); % simple P regulator 

    % clip actual input to avoid constraints braking
    u(i,:) = [40,min(max(value_pump, 0), 100),min(max(value_sp, 0), 100)]; % feed pump pri ident.
    u(i,:)
    u_prev = u(i,:);

    

    pct23.setTag('Pump1',u(i,1)); % feed
    pct23.setTag('Pump2',u(i,2)); % heating media
    pct23.setTag('Heater',u(i,3)); % spiral

    % Wait until 1 second has passed since the start of the iteration
    elapsedTime = toc;
    pause(max(0, 1 - elapsedTime));

end
%% späť - turn on your MPC
% This is initialization of my Kalman filter, you should do it earlyier
py.koopman_mpc.get_xest_koop(y(i,:))
for i = 5*d:12*d
    % Start the timer
    tic;

    % Code to execute within the loop
    disp(['Iteration: ', num2str(i), ' at ', datestr(now)]);
    % take measurements
    y3 = double(pct23.getTag('T4').value);
    y2 = double(pct23.getTag('T2').value);
    y1 = double(pct23.getTag('T1').value);

    y(i,:) = [y1 y2 y3];
    y(i,:)
    pct23.setTag('FSV',1);
    y(i,:) + ys % offset of my current setup to my ident steady state
    u(i,:) = koopman_controll(y(i,:)+ys,u_prev); % MPC calculation here
    u(i,:)
    u_prev = u(i,:);
    % apply the control inputs to device
    pct23.setTag('Pump1',u(i,1)); % feed
    pct23.setTag('Pump2',u(i,2)); % heating media
    pct23.setTag('Heater',u(i,3)); % spiral

    % Wait until 1 second has passed since the start of the iteration
    elapsedTime = toc;
    pause(max(0, 1 - elapsedTime));

end



%terminate(pyenv);
pct23.off();
pct23.setTag('FSV',1);
