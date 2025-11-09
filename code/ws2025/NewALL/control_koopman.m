function u_cmd = control_koopman(y_meas, r_setpoint, opts)
%CONTROL_KOOPMAN Koopman-based MPC with Kalman filtering for pct23 T4 control.
%
%   u_cmd = CONTROL_KOOPMAN(y_meas, r_setpoint) returns the actuation value
%   for Pump2 given the latest measurement y_meas (temperature T4 in °C) and
%   desired setpoint r_setpoint (°C).
%
%   u_cmd = CONTROL_KOOPMAN(y_meas, r_setpoint, opts) allows additional
%   options. Supported fields:
%       - opts.reset : when true, reinitialises internal persistent state.
%
%   Notes:
%       * Internally the function loads Koopman model matrices (A,B,C) and
%         scaling factors from pre-computed datasets.
%       * The function maintains a discrete Kalman filter to estimate the
%         lifted state used by the MPC.
%       * The returned u_cmd is already clipped to the physical limits
%         (0–100 % duty cycle).
%
%   Dependencies:
%       - Requires YALMIP on MATLAB path.
%       - Needs the following data files in the same directory:
%           data/A_wC.npy, data/B_wC.npy, data/C_wC.npy
%           train_data.mat, test_data.mat
%           results_koopman_to_zero.mat (for stored scaling factors)

arguments
    y_meas double {mustBeScalarOrEmpty}
    r_setpoint double {mustBeScalar} = 60
    opts.reset logical = false
end

persistent data

if opts.reset
    data = [];
    u_cmd = NaN;
    return;
end

if isempty(data)
    data = initialise_controller();
end

if isempty(y_meas)
    error('control_koopman:MissingMeasurement', ...
        'Measurement y_meas must be provided unless opts.reset is true.');
end

% Scale measurement and reference
y_scaled = (y_meas - data.x_mean) / data.x_std;
r_scaled = (r_setpoint - data.x_mean) / data.x_std;

if ~data.initialised
    data.x_est = data.C_pinv * y_scaled;
    data.P = data.P0;
    data.u_prev = data.u0;
    data.initialised = true;
else
    % Kalman prediction
    x_pred = data.A * data.x_est + data.B * data.u_prev;
    P_pred = data.A * data.P * data.A' + data.Q_KF;

    % Kalman update
    S = data.C * P_pred * data.C' + data.R_KF;
    K = (P_pred * data.C') / S;
    innovation = y_scaled - data.C * x_pred;
    data.x_est = x_pred + K * innovation;
    data.P = (data.I - K * data.C) * P_pred;
    data.P = (data.P + data.P') / 2;
end

% Solve MPC
params = [data.x_est; r_scaled];
u_scaled = data.controller{params};
u_scaled = full(u_scaled);

% Apply input limits (already enforced in controller, defence in depth)
u_scaled = min(max(u_scaled, data.umin), data.umax);

% Store for next iteration
data.u_prev = u_scaled;

% Descale to physical units
u_cmd = u_scaled * data.u_std + data.u_mean;
u_cmd = min(max(u_cmd, data.u_min_phys), data.u_max_phys);

end

function data = initialise_controller()

data = struct();

% Load Koopman model matrices
data.A = double(readNPY('data/A_wC.npy'));
data.B = double(readNPY('data/B_wC.npy'));
data.C = double(readNPY('data/C_wC.npy'));

data.nx = size(data.A, 1);
data.I = eye(data.nx);

% Scaling factors
scaling = load_scaling();
data.x_mean = scaling.x_mean;
data.x_std = scaling.x_std;
data.u_mean = scaling.u_mean;
data.u_std = scaling.u_std;

% Kalman filter parameters
data.Q_KF = 0.5 * eye(data.nx);
data.R_KF = 0.1;
data.P0 = eye(data.nx);
data.C_pinv = pinv(data.C);

% Input limits in scaled coordinates
data.u_min_phys = 0;
data.u_max_phys = 100;
data.umin = (data.u_min_phys - data.u_mean) / data.u_std;
data.umax = (data.u_max_phys - data.u_mean) / data.u_std;

% Output constraints (safety bounds)
ymin_phys = 0;
ymax_phys = 70;
data.ymin = (ymin_phys - data.x_mean) / data.x_std;
data.ymax = (ymax_phys - data.x_mean) / data.x_std;

% MPC weights and horizon
Qy = 10;
Qu = 1;
N = 40;

% Build YALMIP MPC optimiser with reference as parameter
u = sdpvar(repmat(1,1,N), repmat(1,1,N));
x = sdpvar(repmat(data.nx,1,N+1), repmat(1,1,N+1));
x0_param = sdpvar(data.nx,1);
r_param = sdpvar(1,1);

constraints = [x{1} == x0_param];
objective = 0;

for k = 1:N
    constraints = [constraints, x{k+1} == data.A * x{k} + data.B * u{k}];
    constraints = [constraints, data.umin <= u{k} <= data.umax];
    yk = data.C * x{k};
    constraints = [constraints, data.ymin <= yk <= data.ymax];
    objective = objective + Qy * (yk - r_param)^2 + Qu * (u{k})^2;
end

controller = optimizer(constraints, objective, ...
    sdpsettings('solver','quadprog','verbose',0), ...
    [x0_param; r_param], u{1});

data.controller = controller;

% Initial conditions
data.initialised = false;
data.x_est = zeros(data.nx,1);
data.P = data.P0;
data.u_prev = 0;
data.u0 = 0;

end

function scaling = load_scaling()
% Attempt to load stored scaling; fall back to recomputing from datasets.
scaling = struct();

if exist('results_koopman_to_zero.mat','file')
    tmp = load('results_koopman_to_zero.mat', 'x_mean', 'x_std', 'u_mean', 'u_std');
    scaling.x_mean = tmp.x_mean;
    scaling.x_std = tmp.x_std;
    scaling.u_mean = tmp.u_mean;
    scaling.u_std = tmp.u_std;
else
    % Recompute from train/test datasets
    train = load('train_data.mat','Ytrain','Utrain');
    test = load('test_data.mat','Ytest','Utest');

    Ytrain = train.Ytrain(:);
    Utrain = train.Utrain(:);
    Ytest = test.Ytest(:);
    Utest = test.Utest(:);

    Yall = [Ytrain; Ytest];
    Uall = [Utrain; Utest];

    scaling.x_mean = mean(Yall);
    scaling.x_std = std(Yall);
    scaling.u_mean = mean(Uall);
    scaling.u_std = std(Uall);
end

end

