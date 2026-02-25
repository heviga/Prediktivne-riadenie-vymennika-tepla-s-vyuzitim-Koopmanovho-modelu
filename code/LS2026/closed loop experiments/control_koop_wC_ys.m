hold_loops = 5;        % T4 musí byť na setpointe 5 loopov po sebe
setpoint_preheat = 68; % cieľová teplota pred Koopmanom
filename = 'steps/2611runtime_log_koop8.mat';
[y_T4_pre, y_T2_pre, u_Pump2_pre] = preheat_PI(pct23, setpoint_preheat, Ts, P_spiral, Pump1_const, hold_loops);

T4_init = y_T4_pre(end);


%% ====== Runtime configuration ======
sim_minutes = 5;            % total runtime in minutes
Ts_runtime = Ts;            % reuse device sampling period
num_steps = sim_minutes * 60 / Ts_runtime;
P_spiral = 12;
ys = [59.0676 - 63.8084];   % offset T4 () - xmean z ident
Pump1_const = 50;

% Preallocate logs
y_T4 = zeros(num_steps,1);
y_T2 = zeros(num_steps,1);
u_Pump2 = zeros(num_steps,1);
time_log = strings(num_steps,1);

%% ====== Initialize Koopman MPC ======
%T4_init = double(pct23.getTag('T4').value);
control_koopman(T4_init + ys, struct('reset', true));  % reset s aktuálnou teplotou

% Stabilize system a priori
% pct23.setTag('Pump2', 50);  % u_mean
% pause(1);                   % nech sa systém ustáli

%% ====== Main control loop ======
for k = 1:num_steps
    tic;
    time_log(k) = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');

    % Measurements
    T4 = double(pct23.getTag('T4').value);
    T2 = double(pct23.getTag('T2').value);
    y_T4(k) = T4;
    y_T2(k) = T2;

    % --- Compute Heater command (simple P regulator) ---
    value_sp = min(max(P_spiral * (76 - T2), 0), 100);

    % --- Koopman MPC control for Pump2 ---
    y_offset = T4 + ys;
    u_cmd = control_koopman(y_offset);
    u_cmd = min(max(u_cmd, 0), 100);  % saturácia
    u_Pump2(k) = u_cmd(1);

    % --- Apply control to device ---
    pct23.setTag('Pump2', u_Pump2(k));
    pct23.setTag('Pump1', Pump1_const);
    pct23.setTag('Heater', value_sp);
    pct23.setTag('FSV', 1);

    % --- Display info ---
    fprintf('loop: %d\n', k);
    fprintf('T4 = %.3f °C, akcny zasah = %.3f %%\n', y_T4(k), u_Pump2(k));

    % --- Maintain sampling time ---
    pause(max(0, Ts_runtime - toc));
end

%% ====== Logging and cleanup ======
log_data = table((1:num_steps).', time_log, y_T4, y_T2, u_Pump2, ...
    'VariableNames', {'step','timestamp','T4','T2','Pump2'});


save(filename, 'log_data', 'Pump1_const', 'ys');

disp(['data saved as ', filename]);


pct23.off();
control_koopman(0, struct('reset', true));

%% ====== Mini plot ======
time_vec = (0:num_steps-1) * Ts_runtime;
figure;
subplot(2,1,1);
plot(time_vec, u_Pump2,'r','LineWidth',1.5); grid on;
xlabel('Time (s)'); ylabel('Pump2 (%)'); title('Pump2 command');
subplot(2,1,2);
plot(time_vec, y_T4,'b','LineWidth',1.5); grid on;
xlabel('Time (s)'); ylabel('T4 (°C)'); title('T4 vs Time');
