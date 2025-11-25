% Parameterizable copy of control_strejc_wC_ys.m for master sequencing.
if ~exist('hold_loops','var') || isempty(hold_loops)
    hold_loops = 5;
end

if ~exist('setpoint_preheat','var') || isempty(setpoint_preheat)
    setpoint_preheat = 68;  % cieľová teplota pred Strejc
end

if ~exist('filename','var') || isempty(filename)
    filename = 'steps/runtime_log_strejc10.mat';
end

if ~exist('skip_pct23_off','var') || isempty(skip_pct23_off)
    skip_pct23_off = false;
end

[y_T4_pre, y_T2_pre, u_Pump2_pre] = preheat_PI(pct23, setpoint_preheat, Ts, P_spiral, Pump1_const, hold_loops);

T4_init = y_T4_pre(end);  % použijeme ako počiatočnú teplotu pre Strejc
fprintf('Preheat complete, starting Strejc control from T4 = %.3f °C\n', T4_init);

%% Runtime configuration
sim_minutes = 5;
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

ys = [59.0676 - 64.1115];
Pump1_const = 50;
control_strejc(0, struct('reset', true));

double(pct23.getTag('T4').value); pause(1);
double(pct23.getTag('T2').value); pause(1);
double(pct23.getTag('T1').value);

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
    u_cmd = control_strejc(T4 + ys);
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

%% Logging and cleanup
log_data = table((1:num_steps).', time_log, y_T4, u_Pump2, ...
    'VariableNames', {'step','timestamp','T4','Pump2'});

save(filename, 'log_data', 'Pump1_const', 'ys');

disp(['data saved as ', filename]);

if ~skip_pct23_off
    pct23.off();
end
pct23.setTag('FSV',1);
control_strejc(0, struct('reset', true));

disp('Strejc control runtime finished.');

%% Subplot: time vs u and time vs T4
figure;
time_vec = (0:num_steps-1) * Ts_runtime; % v sekundách
subplot(2,1,1);
plot(time_vec, u_Pump2, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Pump2 command (%)');
title('Time vs Pump2');

subplot(2,1,2);
plot(time_vec, y_T4, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('T4 (°C)');
title('Time vs T4');

