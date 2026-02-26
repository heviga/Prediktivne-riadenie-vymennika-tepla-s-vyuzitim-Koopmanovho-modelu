function u_cmd = control_koopman(y_meas, opts)
%CONTROL_KOOPMAN Real-time Koopman MPC controller with Kalman filter
%   u_cmd = control_koopman(y_meas, r)
%   u_cmd = control_koopman(y_meas, r, struct('reset',true))  -> reset state
% r v nule 58.3377
% P        % = P_{k|k}
% P_pred   % = P_{k|k-1}

persistent A B C D nx ny N Qy Qu Q_KF R_KF ...
    umin umax ymin ymax ...
    x_mean x_std u_mean u_std ...
    controller x_est P initialized

%% === RESET HANDLER ===
if nargin > 2 && isfield(opts,'reset') && opts.reset
    initialized = false;
    u_cmd = 0;
    disp('[control_koopman] State reset.');
    return;
end

%% === INITIALIZATION ===
if isempty(initialized) || ~initialized
    disp('[control_koopman] Initializing Koopman controller...');

    % --- Load model matrices ---
    A = double(readNPY('C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\LS2026\data\A_wC.npy'));
    B = double(readNPY('C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\LS2026\data\B_wC.npy'));
    C = double(readNPY('C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\LS2026\data\C_wC.npy'));
    D = 0;
    nx = size(A,1);
    ny = 1;

    % --- Load scaling data ---
    data = load('results_koopman_to_zero.mat');
    x_mean = data.x_mean;
    u_mean = data.u_mean;
% 
    % odhadneme štandardné odchýlky z uložených dát
    x_std = std(data.y_true_desc(:));
    u_std = std(data.u_cl_desc(:));

    % --- MPC nastavenia ---
    Qy = 10;
    Qu = 1;
    N = 20;

    umin = (20 - u_mean) / u_std;
    umax = (100 - u_mean) / u_std;
    ymin = (0 - x_mean) / x_std;
    ymax = (70 - x_mean) / x_std;

    % --- Kalman filter ---
    Q_KF = 0.5*eye(nx);
    R_KF = 0.1;
    P = eye(nx);

    % --- Vytvorenie MPC riešiteľa (YALMIP) ---
    x = sdpvar(nx,1);
    U = sdpvar(1,N);
    X = sdpvar(nx,N+1);
    x0_param = sdpvar(nx,1);

    constraints = [X(:,1) == x0_param];
    objective = 0;

    for k = 1:N
        constraints = [constraints, X(:,k+1) == A*X(:,k) + B*U(:,k)];
        yk = C*X(:,k);
        constraints = [constraints, umin <= U(:,k) <= umax];
        constraints = [constraints, ymin <= yk <= ymax];
        objective = objective + Qy*(yk)^2 + Qu*(U(:,k))^2;
    end

    controller = optimizer(constraints, objective, sdpsettings('solver','quadprog'), x0_param, U(:,1));

    % --- Initialize state from measurement ---
    y_scaled = (y_meas - x_mean)/x_std;
    x_est = pinv(C) * y_scaled; %\hat{x}_{k|k}

    initialized = true;
    disp('[control_koopman] Controller initialized.');
end

%% === SCALE INPUTS ===
y_scaled = (y_meas - x_mean)/x_std;
%r_scaled = (r - x_mean)/x_std;

%% === KALMAN FILTER ===
% Prediction
x_pred = A*x_est;% = \hat{x}_{k|k-1}
P_pred = A*P*A' + Q_KF;

% Update
S = C*P_pred*C' + R_KF;
K = (P_pred*C')/S;
x_est = x_pred + K*(y_scaled - C*x_pred);
P = (eye(nx)-K*C)*P_pred;
P = (P+P')/2; % symmetrize

%% === MPC CONTROL ===
u_scaled = controller{x_est};
u_cmd = u_scaled * u_std + u_mean;

% --- Saturation (safety clamp) ---
u_cmd = min(max(u_cmd, 0), 100);
end

% control_koopman2(0, 60, struct('reset', true)); % reset kontroléra
% u_cmd = control_koopman2(T4, setpoint_T4);
% control_koopman2(0, 0, struct('reset', true)); %po  meraani
