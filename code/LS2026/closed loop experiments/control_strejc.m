function u_cmd = control_strejc(y_meas, opts)
%CONTROL_STREJC Real-time Strejc MPC controller with Kalman filter
%   u_cmd = control_strejc(y_meas)
%   u_cmd = control_strejc(y_meas, struct('reset',true))  -> reset state

persistent A B C D nx ny N Qy Qu ...
           umin umax ymin ymax ...
           x_mean x_std u_mean u_std ...
           controller x_est P Q_KF R_KF initialized

%% === RESET HANDLER ===
if nargin > 1 && isfield(opts,'reset') && opts.reset
    initialized = false;
    u_cmd = 0;
    disp('[control_strejc] State reset.');
    return;
end

%% === INITIALIZATION ===
if isempty(initialized) || ~initialized
    disp('[control_strejc] Initializing Strejc controller...');

    % --- Load Strejc model parameters ---
    A = 0.98540172;
    B = 0.01640409;
    C = 1;
    D = 0;
    nx = 1;
    ny = 1;

    % --- Load scaling from precomputed data ---
    data = load('results_strejc_to_zero.mat'); % y_true_desc, u_cl_desc
    x_mean = mean(data.y_true_desc(:));
    u_mean = mean(data.u_cl_desc(:));
    x_std = std(data.y_true_desc(:));
    u_std = std(data.u_cl_desc(:));

    % --- MPC settings ---
    N = 20;        
    Qy = 10;       
    Qu = 1;        

    umin = (20 - u_mean)/u_std;
    umax = (100 - u_mean)/u_std;
    ymin = (0 - x_mean)/x_std;
    ymax = (70 - x_mean)/x_std;

    % --- YALMIP optimizer ---
    x = sdpvar(nx,1,'full'); 
    U = sdpvar(1,N,'full');
    X = sdpvar(nx,N+1,'full');
    x0_param = sdpvar(nx,1,'full');

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

    % --- Kalman filter ---
    P = 1;
    Q_KF = 0.5;   % <--- tu
    R_KF = 0.1;   % <--- tu

    % --- Initialize state ---
    y_scaled = (y_meas - x_mean)/x_std;
    x_est = pinv(C) * y_scaled;

    initialized = true;
    disp('[control_strejc] Controller initialized.');
end

%% === SCALE INPUTS ===
y_scaled = (y_meas - x_mean)/x_std;

%% === KALMAN FILTER ===
x_pred = A*x_est;
P_pred = A*P*A' + Q_KF;

S = C*P_pred*C' + R_KF;
K = (P_pred*C')/S;
x_est = x_pred + K*(y_scaled - C*x_pred);
P = (eye(nx) - K*C)*P_pred;
P = (P+P')/2;

%% === MPC CONTROL ===
u_scaled = controller{x_est};
u_cmd = u_scaled * u_std + u_mean;

%% --- Saturation ---
u_cmd = min(max(u_cmd, 0), 100);
end
