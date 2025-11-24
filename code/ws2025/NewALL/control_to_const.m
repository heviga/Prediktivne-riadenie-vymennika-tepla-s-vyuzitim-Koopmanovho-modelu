%% === BASIC CONTROL: Pump2 fixed at 70 (no Koopman) ===

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

%% Runtime configuration
sim_minutes = 7;
setpoint_T4 = 58.3377;    % optional, kept for compatibility with original save
Ts_runtime = Ts;
num_steps = sim_minutes * 60 / Ts_runtime;
P_spiral = 12;

 % ******** PUMP2 KONŠTANTA  ***************************
% Constants
Pump1_const = 50;
Pump2_const = 50;     % ******** PUMP2 KONŠTANTA 70 ********



% Preallocate logs (you requested to keep these)
y_T4 = zeros(num_steps,1);
y_T2 = zeros(num_steps,1);
u_Heater = zeros(num_steps,1);
u_Pump2 = zeros(num_steps,1);
time_log = strings(num_steps,1);

% Reset Strejc controller (if you still use it elsewhere)
control_strejc(0, struct('reset', true));

% quick read to ensure tags are available (same as your working code)
double(pct23.getTag('T4').value)
pause(1)
double(pct23.getTag('T2').value)
pause(1)
double(pct23.getTag('T1').value)

%% Main control loop
for k = 1:num_steps
    tic;
    time_log(k) = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');

    % --- Measurements (same safe read as your working Strejc code) ---
    T4 = double(pct23.getTag('T4').value);
    T2 = double(pct23.getTag('T2').value);

    y_T4(k) = T4;
    y_T2(k) = T2;

    fprintf('T4 = %.3f\n', y_T4(k));

    % --- Compute Heater command (simple P regulator) ---
    value_sp = P_spiral * (76 - T2); % P regulator for spiral
    value_sp = min(max(value_sp, 0), 100);
    u_Heater(k) = value_sp;

    % --- APPLY CONTROLS ---
    u_Pump2(k) = Pump2_const;               % constant Pump2
    pct23.setTag('Pump2', u_Pump2(k));
    pct23.setTag('Pump1', Pump1_const);
    pct23.setTag('Heater', u_Heater(k));
    pct23.setTag('FSV', 1);

    fprintf('loop: %d, Pump2 = %.3f, Heater = %.3f\n', k, u_Pump2(k), u_Heater(k));

    % --- Maintain sampling time ---
    elapsedTime = toc;
    pause(max(0, Ts_runtime - elapsedTime));
end

%% ===============================
%   LOGGING AND CLEANUP
%   ===============================

% log_data = table((1:num_steps).', time_log, y_T4, y_T2, u_Pump2, u_Heater, ...
%     'VariableNames', {'step','timestamp','T4','T2','Pump2','Heater'});
% ys
% % Save (keeps compatibility with earlier variables)
% save('runtime_log_Pump2_70.mat','log_data','setpoint_T4','Pump1_const','ys');

% Turn device off / cleanup
pct23.off();
pct23.setTag('FSV',1);
control_strejc(0, struct('reset', true));

disp('Pump2 = %.4f\n constant control runtime finished.',Pump2_const);
xmean_T4 = mean(y_T4);
xmean_T2 = mean(y_T2);

fprintf('\n=== XMEAN RESULTS ===\n');
fprintf('xmean T4 = %.4f °C\n ', xmean_T4);
fprintf('umean pump2 = %.4f °C\n ', u_mean); % 65.8447
