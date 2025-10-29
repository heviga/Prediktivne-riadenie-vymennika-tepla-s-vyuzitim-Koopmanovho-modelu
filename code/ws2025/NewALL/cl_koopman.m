% spusti prve pre vytvorenie baseline potom strejc a potom coparison
clc,clear all,%close all

% Add path to readNPY function
addpath('../');

% Initialize Python environment for baseline inference
% terminate(pyenv); % Not needed for InProcess mode
pyenv('Version', 'C:\Users\ivadu\AppData\Local\Programs\Python\Python39\python.exe');

% Add Python path for baseline_inference
py.sys.path().append('C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\ws2025\NewALL');

% Initialize baseline inference
py.baseline_inference.init();


% Load Koopman model matrices
A = double(readNPY('data/A_wC.npy'));
B = double(readNPY('data/B_wC.npy'));
C = double(readNPY('data/C_wC.npy'));
D = 0;

% 
% aa= inv(A-eye(size(A)))*B
% cc=C*zeros(10,1)


% Load data
load('train_data.mat');   % Ytrain, Utrain
load('test_data.mat');    % Ytest, Utest

% Flatten
Ytrain = Ytrain(:); Utrain = Utrain(:);
Ytest = Ytest(:);   Utest = Utest(:);

% Full dataset for scaling
Yall = [Ytrain; Ytest];
Uall = [Utrain; Utest];

% Scaling
x_mean = mean(Yall);
x_std = std(Yall);
u_mean = mean(Uall);
u_std = std(Uall);

% Scale test data
Ytest_scaled = (Ytest - x_mean) / x_std;
Utest_scaled = (Utest - u_mean) / u_std;

%% Open-loop Koopman rollout (using Utest)
sim_length = 300;%length(Utest_scaled);
nx = size(A,1);
ny = 1;

x0 = zeros(nx,1); x0(1) = Ytest_scaled(1);%z0
y0 = (50 - x_mean) / x_std; %POC PODMIENKA
x0 = pinv(C)*y0;

x_open = zeros(nx, sim_length+1);
y_open = zeros(ny, sim_length+1);
x_open(:,1) = x0;
y_open(:,1) = C * x0;


%% simulation of ol
for t = 1:sim_length
    x_open(:,t+1) = A * x_open(:,t) + B * Utest_scaled(t);
    y_open(:,t+1) = C * x_open(:,t+1);
end

% Descale open-loop
y_open_desc = y_open * x_std + x_mean;
u_test_desc = Utest_scaled * u_std + u_mean;

%% Closed-loop Koopman control (MPC) with kalman filter
Qy = 10;
Qu = 1;
N = 40;


umin = (20 - u_mean) / u_std;
umax = (100 - u_mean) / u_std;
% ymin = (20 - x_mean) / x_std;
ymin = (0 - x_mean) / x_std;
ymax = (70 - x_mean) / x_std;

r = (60 - x_mean) / x_std;  % Setpoint


% YALMIP setup
u = sdpvar(repmat(1,1,N), repmat(1,1,N));
x = sdpvar(repmat(nx,1,N+1), repmat(1,1,N+1));
x0_param = sdpvar(nx,1);

constraints = []; objective = 0;
constraints = [constraints, x{1} == x0_param];


for k = 1:N
    constraints = [constraints, x{k+1} == A*x{k} + B*u{k}];
    constraints = [constraints, umin <= u{k} <= umax];
    yk = C*x{k};
    constraints = [constraints, ymin <= yk <= ymax];
    objective = objective + Qy*(yk)^2 + Qu*u{k}^2; % r prec ->riadenie do nuly
end

controller = optimizer(constraints, objective, sdpsettings('solver','quadprog'), x0_param, u{1});

% Closed-loop simulation, kvoli kalmanovi je toto 'true'
tic;
x_true = zeros(nx, sim_length+1);
y_true = zeros(ny, sim_length+1);
u_cl = zeros(1, sim_length);
x_est = zeros(nx, sim_length+1);
y_meas = zeros(ny, sim_length+1); %pre sum
%% --- Kalman filter setup ---
Q_KF = 0.5*eye(10);   % process noise cov
R_KF = 0.1;   % measurement noise cov
P = eye(nx);   % initial covariance
meas_noise_std = 0;%sqrt(R_KF);

%initial states
x_true(:,1) = x0;
x_est(:,1) = x0;
y_true(:,1)=y0;
y_meas(:,1) = y_true(:,1) + meas_noise_std*randn(ny,1);


%% simulation of cl

for t = 1:sim_length
    u_cl(:,t) = controller{x_est(:,t)};%z odhadovaneho
    
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
    
    y_meas(:,t+1) = y_true(:,t+1) + meas_noise_std * randn(ny,1);
    
    %kf prediction
    x_pred = A * x_est(:,t) + B * u_cl(:,t); 
    P_pred = A * P * A' + Q_KF; 
    
    %kf update
    S = C*P_pred*C' + R_KF;
    K = (P_pred*C')/S;
    x_est(:,t+1) = x_pred + K*(y_meas(:,t+1) - C*x_pred);
    P = (eye(nx)-K*C)*P_pred;
    P = (P+P')/2; % symmetrize
end

elapsed_time = toc;
fprintf('Koopman MPC simulation time: %.4f seconds\n', elapsed_time);

% Descale closed-loop
y_true_desc = y_true * x_std + x_mean;
y_est_desc  = (C*x_est)*x_std + x_mean;
u_cl_desc = u_cl * u_std + u_mean;
%sum?
%% Plot both open-loop and closed-loop
time = 0:sim_length;

figure;
subplot(3,1,1)
%plot(time, y_open_desc, 'b-', 'LineWidth', 2); hold on;
plot(time, y_true_desc, 'm-', 'LineWidth', 1.5); hold on;
plot(time, y_est_desc, 'b--','LineWidth',2.5);
%plot(time(1:end-1), Ytest, 'k:', 'LineWidth', 1.5);
xlabel('Time step'); ylabel('Output y (°C)');
yline(x_mean)
legend('True output (Baseline)','KF estimate (Koopman)');
title('Koopman MPC + Kalman Filter vs Baseline System');
grid on; grid minor;
ylim([40 70])


subplot(3,1,2)
%stairs(time(1:end-1), u_test_desc, 'r-', 'LineWidth', 2); hold on;
stairs(time(1:end-1), u_cl_desc, 'k--', 'LineWidth', 2);
xlabel('Time step'); ylabel('Input u');
legend( 'MPC control input');
title('Input Comparison');
grid on; grid minor;

subplot(3,1,3)
plot(time, y_true_desc,'m--','LineWidth',1.5); hold on
plot(time, y_est_desc,'b-','LineWidth',1.5);
plot(time, (y_meas*x_std + x_mean),'gx');
xlabel('Time step'); ylabel('y (°C)');
legend('True output','KF estimate','Noisy measurements');
title('Measurement vs KF estimate');
grid on; grid minor;
ylim([40 70])

%% RMSE Calculation
e_open = y_open_desc(1:end-1) - Ytest(:);
e_cl   = y_true_desc(1:end-1) - Ytest(:);

rmse_open = sqrt(mean(e_open(:).^2));
rmse_cl   = sqrt(mean(e_cl(:).^2));
% rmse_cl = sqrt(mean((y_true_desc(:)-60).^2));

fprintf('RMSE (Open-loop)  = %.4f\n', rmse_open);% stupen celzia
fprintf('RMSE (Closed-loop) = %.4f\n', rmse_cl);


save('results_koopman_to_zero.mat', 'y_true_desc', 'y_est_desc', 'u_cl_desc','x_mean','u_mean','u_cl',"y_true");   % From Koopman
save('baseline_reference.mat', 'y_true_desc');  % Save baseline for comparison

