clc,clear all,%close all

% Add path to readNPY function
addpath('../');

% Load Koopman model matrices
A = double(readNPY('data/A_wC_all.npy'));
B = double(readNPY('data/B_wC_all.npy'));
C = double(readNPY('data/C_wC_all.npy'));
D = 0;

% 
% aa= inv(A-eye(size(A)))*B
% cc=C*zeros(10,1)


% Load data
load('data/train_data_ident.mat');   % Ytrain, Utrain
load('data/test_data_ident.mat');    % Ytest, Utest

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

A = 0.97701252;
B = 0.03150894;
C = 1;
D = 0;
%% Open-loop Koopman rollout (using Utest)
sim_length = length(Utest_scaled);
nx = size(A,1);
ny = 1;

x0 = zeros(nx,1); x0(1) = Ytest_scaled(1);%z0
y0 = (65 - x_mean) / x_std;
x0 = pinv(C)*y0;

x_open = zeros(nx, sim_length+1);
y_open = zeros(ny, sim_length+1);
x_open(:,1) = x0;
y_open(:,1) = C * x0;

for t = 1:sim_length
    x_open(:,t+1) = A * x_open(:,t) + B * Utest_scaled(t);
    y_open(:,t+1) = C * x_open(:,t+1);
end

% Descale open-loop
y_open_desc = y_open * x_std + x_mean;
u_test_desc = Utest_scaled * u_std + u_mean;

%% Closed-loop Koopman control (MPC)
Qy = 10;
Qu = 1;
N = 40;

sim_length = 100;
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

% Closed-loop simulation
x_cl = zeros(nx, sim_length+1);

y_cl = zeros(ny, sim_length+1);
u_cl = zeros(1, sim_length);

x_cl(:,1) = x0;
y_cl(:,1)=y0;

for t = 1:sim_length
    u_cl(:,t) = controller{x_cl(:,t)};
    x_cl(:,t+1) = A * x_cl(:,t) + B * u_cl(:,t);
    y_cl(:,t+1) = C * x_cl(:,t+1);
end

% Descale closed-loop
y_cl_desc = y_cl * x_std + x_mean;
u_cl_desc = u_cl * u_std + u_mean;
%sum?
%% Plot both open-loop and closed-loop
time = 0:sim_length;

figure;
subplot(2,1,1)
%plot(time, y_open_desc, 'b-', 'LineWidth', 2); hold on;
plot(time, y_cl_desc, 'm--', 'LineWidth', 2);
hold on
%plot(time(1:end-1), Ytest, 'k:', 'LineWidth', 1.5);
xlabel('Time step'); ylabel('Output y (°C)');
legend( 'Closed-loop (MPC)');
title('Koopman Model Output Comparison');
grid on; grid minor;

subplot(2,1,2)
%stairs(time(1:end-1), u_test_desc, 'r-', 'LineWidth', 2); hold on;
stairs(time(1:end-1), u_cl_desc, 'k--', 'LineWidth', 2);
xlabel('Time step'); ylabel('Input u');
legend( 'MPC control input');
title('Input Comparison');
grid on; grid minor;

%% RMSE Calculation
e_open = y_open_desc(1:end-1) - Ytest(:);
e_cl   = y_cl_desc(1:end-1) - Ytest(:);

rmse_open = sqrt(mean(e_open(:).^2));
rmse_cl   = sqrt(mean(e_cl(:).^2));

fprintf('RMSE (Open-loop)  = %.4f\n', rmse_open);% stupen celzia
fprintf('RMSE (Closed-loop) = %.4f\n', rmse_cl);


save('results_koopman.mat', 'y_cl_desc', 'u_cl_desc');   % From Koopman

