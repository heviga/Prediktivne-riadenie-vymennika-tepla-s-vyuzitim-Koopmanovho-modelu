clc; clear all; %close all

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
Ru = 1;
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
    constraints = [constraints, umin <= u{k} <= umax];
    yk = C * x{k};
    constraints = [constraints, ymin <= yk <= ymax];
    objective = objective + Qy*(yk)^2 + Ru*u{k}^2;  % Control to 0
end

controller = optimizer(constraints, objective, sdpsettings('solver','quadprog'), x0_param, u{1});

%% --- Simulate closed-loop ---
x_cl = zeros(nx, sim_length+1);
y_cl = zeros(ny, sim_length+1);
u_cl = zeros(nu, sim_length);

x_cl(:,1) = x0;

y_cl(:,1) = C * x_cl(:,1);  % 

for t = 1:sim_length
    u_cl(:,t) = controller{x_cl(:,t)};
    x_cl(:,t+1) = A * x_cl(:,t) + B * u_cl(:,t);
    y_cl(:,t+1) = C * x_cl(:,t+1);
end

% Descale
y_cl_desc = y_cl * x_std + x_mean;
u_cl_desc = u_cl * u_std + u_mean;

%% --- Plot closed-loop only ---
time = 0:sim_length;
figure;
subplot(2,1,1)
plot(time, y_cl_desc, 'm--', 'LineWidth', 2); hold on
yline(0, 'r--'); % visual zero line
xlabel('Time step'); ylabel('Output y (°C)');
legend('MPC Output');
title('Strejc Closed-loop ');
grid on;grid minor;
ylim([40 70])

subplot(2,1,2)
stairs(time(1:end-1), u_cl_desc, 'k--', 'LineWidth', 2);
xlabel('Time step'); ylabel('Input u');
title('MPC Input');
grid on;grid minor;

%% --- Save and RMSEC ---
rmse_strejc_to_zero = sqrt(mean((y_cl_desc(:)).^2)); % RMSE to zero °C
fprintf('RMSE (Strejc to 0°C) = %.4f °C\n', rmse_strejc_to_zero);

save('results_strejc_to_zero.mat', 'y_cl_desc', 'u_cl_desc');
