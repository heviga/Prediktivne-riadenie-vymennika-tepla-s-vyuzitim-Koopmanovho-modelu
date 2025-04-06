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

load('data/scaled_data.mat')
load('data/unscaled_data.mat')

u_scaled = u_scaled(:); 
x_scaled = x_scaled(:); 

x_mean = 58.3152398;
x_std = 9.07091605;

u_mean = 54.6108889572;
u_std = 27.6293198476;


%% 1. strejc a mpc na strejca 
% tiez ho naskalovat najprv

% Discrete-time Strejc model
A = 0.9770;
B = 0.0319;
C = 1;
D = 0;

Ts = 1;             % Sampling time
nx = 1;             % Number of states
nu = 1;             % Number of inputs
ny = 1;             % Number of outputs

sim_steps = length(x_scaled);
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
sim_steps = length(x_scaled);
x_sim = zeros(nx, sim_steps+1);
u_sim = zeros(nu, sim_steps);
y_sim = zeros(ny, sim_steps+1);

x_sim(:,1) = x_scaled(1); % Initial condition
y_sim(:,1) = C * x_sim(:,1);

for t = 1:sim_steps
    % Get optimal input
    u_opt = controller{x_sim(:,t)};
    u_sim(:,t) = u_opt;

    % Apply input to system
    x_sim(:,t+1) = A * x_sim(:,t) + B * u_sim(:,t);
    y_sim(:,t+1) = C * x_sim(:,t+1);
end
%% plot

u_descaled = u_sim * u_std + u_mean;
y_descaled = y_sim * x_std + x_mean;

time = 0:sim_steps;

figure;
subplot(2,1,1)
plot(time, y_descaled, 'LineWidth', 2);
hold on; yline(x_mean + x_std * r, '--r', 'Setpoint'); %x_mean + x_std * r
ylabel('Output y'); grid on; title('MPC Response - descaled');
ylim([40 80])
xlim([0 length(y_descaled)])

subplot(2,1,2)
stairs(0:sim_steps-1, u_descaled, 'LineWidth', 2);
ylabel('Input u'); xlabel('Time step'); grid on;
xlim([0 length(y_descaled)])


%%