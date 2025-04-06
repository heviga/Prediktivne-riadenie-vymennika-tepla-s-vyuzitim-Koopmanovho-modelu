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

load('train_data_ident.mat');  % Ytrain, Utrain (unscaled)
load('test_data_ident.mat');

Ytrain = Ytrain(:);
Utrain = Utrain(:);
Ytest = Ytest(:);
Utest = Utest(:);

x_mean = mean(Ytrain);
x_std = std(Ytrain);
u_mean = mean(Utrain);
u_std = std(Utrain);

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
    constraints = [constraints, x{k+1} == A*x{k} + B*u{k}];

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
% Simulation
sim_steps = length(u_scaled_test);

x_strejc = zeros(nx, sim_steps+1);
y_strejc = zeros(ny, sim_steps+1);

x_sim(:,1) = x_scaled(1); % Initial condition
y_sim(:,1) = C * x_sim(:,1);

% Initial condition from test data
x_strejc(:,1) = x_scaled_test(1);
y_strejc(:,1) = C * x_strejc(:,1);

for t = 1:sim_steps
    x_strejc(:,t+1) = A * x_strejc(:,t) + B * u_scaled_test(t);
    y_strejc(:,t+1) = C * x_strejc(:,t+1);
end

%% plot
u_descaled = u_scaled_test * u_std + u_mean;
y_descaled = y_strejc * x_std + x_mean;
time = 0:sim_steps;

figure;
subplot(2,1,1)
plot(time, y_descaled, 'LineWidth', 2); % Koopman prediction (or Strejc)
hold on;
plot(time(1:end-1), Ytest, '--k', 'LineWidth', 1.5); % Ground truth
xlabel('Time step'); ylabel('Output y (°C)');
legend('Strejc Predicted', 'True Test Output');
title('Strejc Model Prediction (Test Data)');
grid on;

subplot(2,1,2)
stairs(time(1:end-1), u_descaled, 'LineWidth', 2);
xlabel('Time step'); ylabel('Input u');
title('Test Input (Descaled)');
grid on;

%%