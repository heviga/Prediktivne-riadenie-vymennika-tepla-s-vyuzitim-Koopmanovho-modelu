
%#ucka potrebujeme descalovat u = u_scaled * u_std + u_mean;
%% 
close all, 
clear all
%%
load('../data/train_data.mat');  % Ytrain, Utrain (unscaled)
load('../data/test_data.mat');

Ytrain = Ytrain(:);
Utrain = Utrain(:);
Ytest = Ytest(:);
Utest = Utest(:);

Yall = [Ytrain; Ytest];
Uall = [Utrain; Utest];

y_mean = mean(Yall);
x_std = std(Yall);
u_mean = mean(Uall);
u_std = std(Uall);

% Scale test data
y_scaled_test = (Ytest - y_mean) / x_std;
u_scaled_test = (Utest - u_mean) / u_std;


%  59.0676 - z ident
% u_mean = mean(u); % 65.8447 - z ident


%% 1. strejc a mpc na strejca 
% tiez ho naskalovat najprv
K = 1.1237;
Ts = 1;
tau = 68;
% Discrete-time Strejc model
A = exp(-Ts/tau); %0.985401721021654
B = K*(1 - A);%0.016404086087968
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
%r = (70 - y_mean) / x_std;               % Setpoint

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
    objective = objective + Qy*(yk)^2 + Ru*u{k}^2;
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

x_open(:,1) = y_scaled_test(1);
y_open(:,1) = C * x_open(:,1);

for t = 1:sim_steps
    x_open(:,t+1) = A * x_open(:,t) + B * u_scaled_test(t);
    y_open(:,t+1) = C * x_open(:,t+1);
end


%% Closed-loop MPC simulation
x_mpc = zeros(nx, sim_steps+1);
y_mpc = zeros(ny, sim_steps+1);
u_mpc = zeros(nu, sim_steps);
x_mpc(:,1) = y_scaled_test(1);% initial


for t = 1:sim_steps
    u_mpc(:,t) = controller{x_mpc(:,t)};
    x_mpc(:,t+1) = A * x_mpc(:,t) + B * u_mpc(:,t);
    y_mpc(:,t+1) = C * x_mpc(:,t+1);
end

%% Descale
time = 0:sim_steps;
u_open_desc = u_scaled_test * u_std + u_mean;
y_open_desc = y_open * x_std + y_mean;
u_mpc_desc = u_mpc * u_std + u_mean;
y_mpc_desc = y_mpc * x_std + y_mean;
y_true = Ytest;

%% Open-loop model accuracy (RMSE)
% Align lengths
y_model_ol = y_open_desc(2:end);   % model output (skip initial state)
y_meas = y_true(:);                % measured output

rmse_ol = sqrt(mean((y_model_ol - y_meas).^2));

fprintf('Open-loop RMSE (Strejc model): %.3f °C\n', rmse_ol);

%% Overall open-loop RMSE (entire dataset)
% Use descaled signals in physical units

y_model = y_open_desc(2:end);   % model output
y_model = y_model(:);
y_meas  = y_true(:);            % measured output

rmse_openloop = sqrt(mean((y_model - y_meas).^2));

fprintf('Overall open-loop RMSE (Strejc): %.3f °C\n', rmse_openloop);


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
split_idx = length(Utrain);

% Split input
U_train = Uall(1:split_idx);
U_test = Uall(split_idx+1:end);
u_scaled_train = u_scaled_all(1:split_idx);
u_scaled_test = u_scaled_all(split_idx+1:end);

figure;

% Use tiledlayout
t = tiledlayout(1,1);
t.TileSpacing = 'compact';
t.Padding = 'compact';

% === Top plot (true input)
nexttile;
p1 = plot(1:split_idx, U_train, 'b-', 'LineWidth', 1.5); hold on;
p2 = plot(split_idx+1:length(Uall), U_test, 'r-', 'LineWidth', 1.5);
xline(split_idx, 'k', 'LineWidth', 2);
ylabel('Input u (\%)');
xlabel('Time (s)');
title('Strejc Input Signal');
grid on;
xlim([0 length(Uall)]);

% === Bottom plot (scaled input)
% nexttile;
% plot(1:split_idx, u_scaled_train, 'b-', 'LineWidth', 1.5); hold on;
% plot(split_idx+1:length(Uall), u_scaled_test, 'r-', 'LineWidth', 1.5);
% xline(split_idx, 'k', 'LineWidth', 2);
% ylabel('Scaled Input u');
% xlabel('Time (s)');
% title('Scaled Input Signal');
% grid on;
% xlim([0 length(Uall)]);

% === Shared legend
lgd = legend([p1 p2], {'Training Data', 'Testing Data'}, ...
    'Orientation', 'horizontal', ...
    'Location', 'southoutside');


% Save figure
%saveas(gcf, 'figs\input_changes.png');

%% Save for later plotting
%save('strejc_ol_data.mat', ...
 %   'y_open_desc', 'y_true', 'time', 'u_open_desc'); % iba test - 1000
 %   samples

min(u_scaled_all)
max(u_scaled_all)
