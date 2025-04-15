% Steady-state gain K ≈ 1.3058
% Time constant τ ≈ 45 samples

% MATLAB mean(x): 58.3152397954
% MATLAB std(x): 9.0723114709

% MATLAB mean(u): 54.6108889572
% MATLAB std(u): 27.6293198476


% Discrete A matrix: [[0.97701252]]v  
% Discrete B matrix: [[0.03150894]]
% Discrete C matrix: [[1]]
% Discrete D matrix: [[0]]

%#ucka potrebujeme descalovat u = u_scaled * u_std + u_mean;
%% load
load('train_data_ident.mat');  % Ytrain, Utrain (unscaled)
load('test_data_ident.mat');
% 
% Ytrain = Ytrain(:);
% Utrain = Utrain(:);
% Ytest = Ytest(:);
% Utest = Utest(:);
% 
% x_mean = mean(Ytrain);
% x_std = std(Ytrain);
% u_mean = mean(Utrain);
% u_std = std(Utrain);
% 
% % Scale test data
% 
% x_scaled_test = (Ytest - x_mean) / x_std;
% u_scaled_test = (Utest - u_mean) / u_std;
% x_mean = 58.3152398;
% x_std = 9.07091605;
% 
% u_mean = 54.6108889572;
% u_std = 27.6293198476;

%% scale on full dataset
Y_all = [Ytrain(:); Ytest(:)];
U_all = [Utrain(:); Utest(:)];

x_mean = mean(Y_all);
x_std = std(Y_all);
u_mean = mean(U_all);
u_std = std(U_all);

% Save for reference
fprintf('x_mean = %.10f\nx_std = %.10f\n', x_mean, x_std);
fprintf('u_mean = %.10f\nu_std = %.10f\n', u_mean, u_std);

%% 1. strejc a mpc na strejca 
% tiez ho naskalovat najprv

% Discrete-time Strejc model
A = 0.97701252;
B = 0.03150894;
C = 1;
D = 0;

Ts = 1;             % Sampling time
nx = 1;             % Number of states
nu = 1;             % Number of inputs
ny = 1;             % Number of outputs

%% yalmip
% Horizon
N = 20;

% Variables
u = sdpvar(repmat(nu,1,N), repmat(1,1,N));
x = sdpvar(repmat(nx,1,N+1), repmat(1,1,N+1));

% Parameters
x0 = sdpvar(nx,1);   % Initial condition
r = 1;               % Setpoint

% Cost weights
Qy = 10;
Ru = 1;

% Input bounds
umax = 1;
umin = -1;

% Constraints and objective
constraints = [];
objective = 0;

for k = 1:N
    % Dynamics constraint
    constraints = [constraints, x{k+1} == A*x{k} + B*u{k}];%cista predikcia

    % Input constraint
    constraints = [constraints, umin <= u{k} <= umax];

    % Cost (tracking + regularization)
    yk = C*x{k};      % Output
    objective = objective + Qy*(yk - r)^2 + Ru*u{k}^2;
end

% Options for the solver
options = sdpsettings('verbose', 0, 'solver', 'quadprog');

% Build optimizer
controller = optimizer(constraints, objective, options, x0, u{1});

%% Generate random inputs for prediction
sim_time = 2000;
step_interval = 250;
num_steps = ceil(sim_time / step_interval);

%random input
rng(42);  % For reproducibility
u_step_vals = ceil(rand(num_steps, 1) * 100 / 10) * 10; %0-100
u_scaled = (u_step_vals - u_mean) / u_std;
u_scaled = repelem(u_scaled, step_interval);
u_scaled = u_scaled(1:sim_time); 


% Simulate Strejc system with random inputs
x_sim = zeros(nx, sim_time+1);
y_sim = zeros(ny, sim_time+1);
x_sim(:,1) = (55 - x_mean)/x_std;  % Initial condition (scaled) 55 akoze prva


for t = 1:sim_time
    x_sim(:,t+1) = A * x_sim(:,t) + B * u_scaled(t);
    y_sim(:,t+1) = C * x_sim(:,t+1);
end

% Descale
y_desc = y_sim * x_std + x_mean;
u_desc = u_scaled * u_std + u_mean;

%% closed loop

x_mpc = zeros(nx, sim_time+1);
y_mpc = zeros(ny, sim_time+1);
u_mpc = zeros(nu, sim_time);
x_mpc(:,1) = (55 - x_mean)/x_std;  % Initial condition (same as before)

for t = 1:sim_time
    u_mpc(:,t) = controller{x_mpc(:,t)};  % compute optimal control
    x_mpc(:,t+1) = A * x_mpc(:,t) + B * u_mpc(:,t);
    y_mpc(:,t+1) = C * x_mpc(:,t+1);
end

% Descale
u_mpc_desc = u_mpc * u_std + u_mean;
y_mpc_desc = y_mpc * x_std + x_mean;

%% Plot
time = 0:sim_time;

figure;
subplot(2,1,1)
plot(time, y_desc, 'b-', 'LineWidth', 2); 
hold on;
plot(time, y_mpc_desc, 'm--', 'LineWidth', 2);
xlabel('Time step'); ylabel('Output y (°C)');
legend('Random Input', 'MPC Control');
title('Strejc Model Prediction with Random Inputs and MPC');
grid on; grid minor;

subplot(2,1,2)
stairs(time(1:end-1), u_desc, 'r-', 'LineWidth', 2); hold on
stairs(time(1:end-1), u_mpc_desc, 'k--', 'LineWidth', 2);
xlabel('Time step'); ylabel('Input u');
legend('Random Input', 'MPC Control');
grid on; grid minor;
%x strejc y strejc

%vykreslit riadenie do ss, 

%% RMSE Calculation (to reference setpoint)
y_ref = ones(size(y_desc)) * 60;       % reference output in °C
rmse_open = sqrt(mean((y_desc - y_ref).^2));
rmse_cl   = sqrt(mean((y_mpc_desc - y_ref).^2));

fprintf('RMSE (Open-loop)  = %.4f °C\n', rmse_open);
fprintf('RMSE (Closed-loop) = %.4f °C\n', rmse_cl);

save('results_strejc.mat', 'y_mpc_desc', 'u_mpc_desc');  % From Strejc

%%