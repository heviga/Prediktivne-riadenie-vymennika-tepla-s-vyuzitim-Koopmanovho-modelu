clear all; %close all
clc;

%% ===== Load data (len kvôli škálovaniu) =====
load('train_data.mat');  % Ytrain, Utrain
load('test_data.mat');   % Ytest, Utest

Ytrain = Ytrain(:); Utrain = Utrain(:);
Ytest  = Ytest(:);  Utest  = Utest(:);

Yall = [Ytrain; Ytest];
Uall = [Utrain; Utest];

x_mean = mean(Yall);
x_std  = std(Yall);
u_mean = mean(Uall);
u_std  = std(Uall);

%% ===== Load BASELINE reference (už hotová trajektória) =====
Sbase = load('baseline_reference.mat');   % musi obsahovat aspon y_true_desc
y_true_desc_full = Sbase.y_true_desc(:);  % v °C

% dĺžka simulácie podľa baseline
sim_length = min(300, length(y_true_desc_full)-1);
time = 0:sim_length;

% orež baseline na sim_length
y_true_desc = y_true_desc_full(1:sim_length+1);

% pre KF + MPC pracujeme v scaled doméne
y_true = (y_true_desc - x_mean) / x_std;  % scaled baseline "plant output"

%% ===== Strejc model (discrete) =====
% (hodnoty máš z identifikácie) :contentReference[oaicite:0]{index=0}
A = 0.98540172;
B = 0.01640409;
C = 1;
D = 0;

nx = 1; ny = 1; nu = 1;

%% ===== MPC setup (ako si mala) =====
Qy = 10;
Qu = 1;
N  = 20;

r = (0 - x_mean) / x_std;     % ponechávam tvoje nastavenie (riadenie "do nuly")
umin = (20  - u_mean) / u_std;
umax = (100 - u_mean) / u_std;
ymin = (0   - x_mean) / x_std;
ymax = (70  - x_mean) / x_std;

u = sdpvar(repmat(1,1,N), repmat(1,1,N));
x = sdpvar(repmat(nx,1,N+1), repmat(1,1,N+1));
x0_param = sdpvar(nx,1);

constraints = [x{1} == x0_param];
objective = 0;

for k = 1:N
    constraints = [constraints, x{k+1} == A*x{k} + B*u{k}];
    constraints = [constraints, umin <= u{k} <= umax];
    yk = C*x{k};
    constraints = [constraints, ymin <= yk <= ymax];
    objective = objective + Qy*(yk)^2 + Qu*u{k}^2;
end

controller = optimizer(constraints, objective, ...
    sdpsettings('solver','quadprog','verbose',0), x0_param, u{1});

%% ===== Kalman filter (scaled) =====
Q_kalman = 0.5;
R_kalman = 0.1;
P0 = 1;
P  = P0;

meas_noise_std = 0; % ak chceš šum, daj sqrt(R_kalman)

%% ===== Simulation: Strejc estimator + MPC, plant = LOADED baseline =====
x_est  = zeros(nx, sim_length+1);
u_cl   = zeros(nu, sim_length);
y_meas = zeros(ny, sim_length+1);

% inicializácia od 1. bodu baseline
x_est(:,1)  = y_true(1);
y_meas(:,1) = y_true(1) + meas_noise_std*randn(ny,1);

for t = 1:sim_length
    % MPC na základe odhadu
    u_cl(:,t) = controller{x_est(:,t)};

    % "Plant output" je baseline (už hotové) -> len prečítaj
    y_meas(:,t+1) = y_true(t+1) + meas_noise_std*randn(ny,1);

    % KF predict
    x_pred = A*x_est(:,t) + B*u_cl(:,t);
    P_pred = A*P*A' + Q_kalman;

    % KF update
    S = C*P_pred*C' + R_kalman;
    K = (P_pred*C')/S;
    x_est(:,t+1) = x_pred + K*(y_meas(:,t+1) - C*x_pred);
    P = (eye(nx) - K*C)*P_pred;
    P = (P+P')/2;
end

%% ===== Descale =====
y_est_desc = x_est * x_std + x_mean;
u_cl_desc  = u_cl * u_std + u_mean;
y_meas_desc = y_meas * x_std + x_mean;

%% ===== Plot =====
figure;

subplot(3,1,1)
plot(time, y_true_desc, 'm-', 'LineWidth', 1.5); hold on
plot(time, y_est_desc,  'b--','LineWidth', 2.5);
yline(x_mean);
xlabel('Time step'); ylabel('Output y (°C)');
legend('True output (Loaded Baseline)','KF Estimate (Strejc)');
title('Strejc MPC + KF (Plant = loaded baseline)');
grid on; grid minor;
ylim([40 70])

subplot(3,1,2)
stairs(time(1:end-1), u_cl_desc, 'k--', 'LineWidth', 2);
xlabel('Time step'); ylabel('Input u');
title('Strejc MPC input');
grid on; grid minor;

subplot(3,1,3)
plot(time, y_true_desc,'m--','LineWidth',1.5); hold on
plot(time, y_est_desc,'b-','LineWidth',1.5);
plot(time, y_meas_desc,'gx');
xlabel('Time step'); ylabel('y (°C)');
legend('Baseline true','KF estimate','Noisy measurements');
title('Measurements vs KF');
grid on; grid minor;
ylim([40 70])

%% ===== Save =====
save('results_strejc_loaded_baseline.mat', ...
    'y_true_desc','y_est_desc','u_cl_desc','x_mean','x_std','u_mean','u_std');
