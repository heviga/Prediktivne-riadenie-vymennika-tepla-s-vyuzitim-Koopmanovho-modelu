%clc; 
clear all; %close all

% Configure Python environment for baseline inference
pyenv('Version', 'C:\Users\ivadu\AppData\Local\Programs\Python\Python39\python.exe');

% Add Python path for baseline_inference
py.sys.path().append('C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\ws2025');

% Initialize baseline inference
py.baseline_inference.init();

% Load data
load('train_data_ident.mat');  % Ytrain, Utrain
load('test_data_ident.mat');   % Ytest, Utest

% Flatten
Ytrain = Ytrain(:);
Utrain = Utrain(:);
Ytest = Ytest(:);
Utest = Utest(:);

% Full dataset for scaling
Yall = [Ytrain; Ytest];
Uall = [Utrain; Utest];

x_mean = mean(Yall);
x_std = std(Yall);
u_mean = mean(Uall);
u_std = std(Uall);

% Discrete Strejc model parameters
A = 0.97701252;
B = 0.03150894;
C = 1;
D = 0;

nx = 1;
ny = 1;
nu = 1;
Ts = 1;
sim_length = 150;

% Scale initial condition
x0 = (50 - x_mean) / x_std;   % Start from 50°C

%% --- MPC setup for control to 0°C ---
Qy = 10;
Qu = 1;
N = 40;

r = (0 - x_mean) / x_std;   % Setpoint = 0°C (scaled)
umin = (20 - u_mean) / u_std;
umax = (100 - u_mean) / u_std;
ymin = (0 - x_mean) / x_std;
ymax = (70 - x_mean) / x_std;

% YALMIP vars
u = sdpvar(repmat(1,1,N), repmat(1,1,N));
x = sdpvar(repmat(nx,1,N+1), repmat(1,1,N+1));
x0_param = sdpvar(nx,1);

constraints = [x{1} == x0_param];
objective = 0;

for k = 1:N
    constraints = [constraints, x{k+1} == A*x{k} + B*u{k}];
    constraints = [constraints, umin <= u{k} <= umax];%na zasahy
    yk = C * x{k};
    constraints = [constraints, ymin <= yk <= ymax];%na stavy
    objective = objective + Qy*(yk)^2 + Qu*u{k}^2;  % Control to 0
end

controller = optimizer(constraints, objective, sdpsettings('solver','quadprog'), x0_param, u{1});

%% --- Kalman filter parameters (scaled domain) ---
% Tieto hodnoty môžeš doladiť podľa reality:
Q_kalman = 1e-5;    % process noise covariance (small: model veríme)
R_kalman = 1e-2;    % measurement noise covariance (väčšie -> menej dôvery v meranie)
P0 = 1;             % počiatočná kovariancia
meas_noise_std = sqrt(R_kalman); % pre simulované merania
%% --- Simulate closed-loop ---
x_true = zeros(nx, sim_length+1);
y_true = zeros(ny, sim_length+1); %tvarime sa ze je to merane
u_cl = zeros(nu, sim_length);
x_est = zeros(nx, sim_length+1);   % Kalman estimated state history
y_meas = zeros(ny, sim_length+1); % pre sum
P=P0;

%initial true, estimated state
x_true(:,1) = x0;
x_est(:,1) = x0;

y_true(:,1) = C * x_true(:,1);  % 
y_meas(:,1) = y_true(:,1) + meas_noise_std * randn(ny,1);


for t = 1:sim_length
    u_cl(:,t) = controller{x_est(:,t)};%toto zistit ci sem ide u_cl/u_est

    % === BASELINE INFERENCE INTEGRATION ===
    % Use baseline model for true system dynamics
    if t == 1
        % Initialize baseline model with current state
        py.baseline_inference.get_x(y_true(:,t));
    end
    
    % Get measurement from baseline model (true system)
    y_baseline = py.baseline_inference.y_plus(u_cl(:,t));
    y_baseline_array = double(y_baseline);
    y_true(:,t+1) = y_baseline_array(1); % Extract scalar value
  
    %noised
    y_meas(:,t+1) = y_true(:,t+1) + meas_noise_std * randn(ny,1);


    %tu filter x_KF=...
    x_pred = A * x_est(:,t) + B * u_cl(:,t);         % x_{t+1|t}
    P_pred = A * P * A' + Q_kalman;                  % P_{t+1|t}
    
    %vykreslit x_KF, x_cl
     % --- Kalman gain and update using measurement y_meas(:,t+1) ---
    S = C * P_pred * C' + R_kalman;                  % innovation covariance (scalar)
    K = (P_pred * C') / S;                           % Kalman gain (nx x ny)
    x_est(:,t+1) = x_pred + K * (y_meas(:,t+1) - C * x_pred); %x est
    P = (eye(nx) - K * C) * P_pred;
    
    % (optional) keep P positive definite numerically
    P = (P + P')/2;
end

% Descale
y_true_desc = y_true * x_std + x_mean; % cl
u_cl_desc = u_cl * u_std + u_mean;
y_est_desc = x_est * x_std + x_mean;  %kf estimation (using Strejc scaling)
%% --- Plot closed-loop only ---
time = 0:sim_length;
figure;
subplot(3,1,1)
plot(time, y_true_desc, 'm-', 'LineWidth', 1.5); hold on
plot(time, y_est_desc, 'b--', 'LineWidth', 2.5);%observer
xlabel('Time step'); ylabel('Output y (°C)');
legend('True output (Baseline)','KF Estimate (Strejc)');
title('Strejc MPC + Kalman Filter vs Baseline System');
grid on;grid minor;
ylim([40 70])

subplot(3,1,2)
stairs(time(1:end-1), u_cl_desc, 'k--', 'LineWidth', 2);
xlabel('Time step'); ylabel('Input u');
title('MPC Input');
grid on;grid minor;

subplot(3,1,3)
% plot measurement (noisy) and true
plot(time, y_true_desc, 'm--', 'LineWidth', 1.5); hold on
plot(time, y_est_desc, 'b-', 'LineWidth', 1.5);
plot(time, (y_meas * x_std + x_mean), 'gx'); % noisy measurements (descaled)
xlabel('Time step'); ylabel('Output y (°C)');
legend('Plant true', 'KF estimate', 'Noisy measurements');
title('Measurements vs KF');
grid on;grid minor;
ylim([40 70])


%% --- Save and RMSEC ---
rmse_strejc_to_zero = sqrt(mean((y_true_desc(:)).^2)); % RMSE to zero °C
fprintf('RMSE (Strejc to 0°C) = %.4f °C\n', rmse_strejc_to_zero);

save('results_strejc_to_zero.mat', 'y_true_desc', 'y_est_desc', 'u_cl_desc');
save('baseline_reference_strejc.mat', 'y_true_desc');  % Save Strejc baseline for comparison
