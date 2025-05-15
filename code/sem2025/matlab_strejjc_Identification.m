% Steady-state gain K ≈ 1.3058
% Time constant τ ≈ 45 samples

% MATLAB mean(x): 58.3152397954
% MATLAB std(x): 9.0723114709

% MATLAB mean(u): 54.6108889572
% MATLAB std(u): 27.6293198476


% Discrete A matrix: [[0.97701252]]
% Discrete B matrix: [[0.03150894]]
% Discrete C matrix: [[1]]
% Discrete D matrix: [[0]]

%#ucka potrebujeme descalovat u = u_scaled * u_std + u_mean;
%% 
%close all, 
clear all
%%
load('train_data_ident.mat');  % Ytrain, Utrain (unscaled)
load('test_data_ident.mat');

Ytrain = Ytrain(:);
Utrain = Utrain(:);
Ytest = Ytest(:);
Utest = Utest(:);

Yall = [Ytrain; Ytest];
Uall = [Utrain; Utest];

x_mean = mean(Yall);
x_std = std(Yall);
u_mean = mean(Uall);
u_std = std(Uall);

% Scale test data
x_scaled_test = (Ytest - x_mean) / x_std;
u_scaled_test = (Utest - u_mean) / u_std;

% x_mean = 58.3152398;
% x_std = 9.07091605;
% 
% u_mean = 54.6108889572;
% u_std = 27.6293198476;


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
N = 40;

% Variables
u = sdpvar(repmat(nu,1,N), repmat(1,1,N));
x = sdpvar(repmat(nx,1,N+1), repmat(1,1,N+1));

% Parameters
x0 = sdpvar(nx,1);   % Initial condition
r = (70 - x_mean) / x_std;               % Setpoint

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

%%
% Simulation open loop
sim_steps = length(u_scaled_test);

x_open = zeros(nx, sim_steps+1);
y_open = zeros(ny, sim_steps+1);

x_open(:,1) = x_scaled_test(1);
y_open(:,1) = C * x_open(:,1);

for t = 1:sim_steps
    x_open(:,t+1) = A * x_open(:,t) + B * u_scaled_test(t);
    y_open(:,t+1) = C * x_open(:,t+1);
end


%% Closed-loop MPC simulation
x_mpc = zeros(nx, sim_steps+1);
y_mpc = zeros(ny, sim_steps+1);
u_mpc = zeros(nu, sim_steps);
x_mpc(:,1) = x_scaled_test(1);% initial


for t = 1:sim_steps
    u_mpc(:,t) = controller{x_mpc(:,t)};
    x_mpc(:,t+1) = A * x_mpc(:,t) + B * u_mpc(:,t);
    y_mpc(:,t+1) = C * x_mpc(:,t+1);
end

%% Descale
time = 0:sim_steps;
u_open_desc = u_scaled_test * u_std + u_mean;
y_open_desc = y_open * x_std + x_mean;
u_mpc_desc = u_mpc * u_std + u_mean;
y_mpc_desc = y_mpc * x_std + x_mean;
y_true = Ytest;


%% plot
time = 0:sim_steps;

figure;
subplot(2,1,1)
plot(time, y_open_desc, 'b-', 'LineWidth', 2); hold on
plot(time(1:end-1), y_true, 'k:', 'LineWidth', 1.5);
xlabel('Time step'); ylabel('Output y (°C)');
legend('Open-loop (Strejc)', 'True Output');
title('Output comparison');
grid on;

subplot(2,1,2)
stairs(time(1:end-1), u_open_desc, 'r-', 'LineWidth', 2); hold on;
xlabel('Time step'); ylabel('Input u');
legend('Open-loop Input');
title('Input Comparison');
grid on; grid minor;

%% save inputs

u_scaled_all = (Uall - u_mean) / u_std;
split_idx = length(Uall) - 2000;
figure;

subplot(2,1,1)
plot(Uall, 'b-', 'LineWidth', 1.5);hold on;
xline(split_idx, 'k','LineWidth',2.5);
ylabel('True Input u (\%)');
title('True Input Signal');
grid on;
xlim([0 length(Uall)])

subplot(2,1,2)
plot(u_scaled_all, 'r--', 'LineWidth', 1.5);hold on;
xline(split_idx, 'k','LineWidth',2.5);
ylabel('Scaled Input u');
xlabel('Sample');
title('Scaled Input Signal');
grid on;
xlim([0 length(Uall)])

% Save figure
saveas(gcf, 'C:\Users\ivadu\Desktop\8.semestrik\vymennik\prez\input_comparison.png');
%% Save for later plotting
%save('strejc_open_loop_comparison_data.mat', ...
%    'y_open_desc', 'y_true', 'time', 'u_open_desc');

min(u_scaled_all)
max(u_scaled_all)
