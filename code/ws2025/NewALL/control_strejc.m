function u_cmd = control_strejc(y_meas, r_setpoint, opts)
%CONTROL_STREJC MPC controller with Strejc model and Kalman filter.
%
%   u_cmd = CONTROL_STREJC(y_meas, r_setpoint) returns the Pump2 command
%   (in %) that drives the outlet temperature T4 toward r_setpoint (°C).
%
%   u_cmd = CONTROL_STREJC(y_meas, r_setpoint, opts) accepts:
%       opts.reset  - logical flag; when true, internal persistent states
%                    are reinitialized and NaN is returned.
%
%   The function keeps persistent data with the Strejc model, scaling
%   factors, Kalman filter covariance, and a pre-built YALMIP optimiser.

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
    error('control_strejc:MissingMeasurement', ...
        'Measurement y_meas must be provided unless opts.reset is true.');
end

% Scale measurement and reference
y_scaled = (y_meas - data.x_mean) / data.x_std;
r_scaled = (r_setpoint - data.x_mean) / data.x_std;

if ~data.initialised
    data.x_est = y_scaled; % C = 1 => state equals output in scaled domain
    data.P = data.P0;
    data.u_prev = data.u0;
    data.initialised = true;
else
    % 1D Kalman filter
    x_pred = data.A * data.x_est + data.B * data.u_prev;
    P_pred = data.A * data.P * data.A' + data.Q_KF;

    S = data.C * P_pred * data.C' + data.R_KF;
    K = (P_pred * data.C') / S;
    innovation = y_scaled - data.C * x_pred;
    data.x_est = x_pred + K * innovation;
    data.P = (1 - K * data.C) * P_pred;
    data.P = (data.P + data.P') / 2;
end

% MPC control law
params = [data.x_est; r_scaled];
u_scaled = data.controller{params};
u_scaled = full(u_scaled);
u_scaled = min(max(u_scaled, data.umin), data.umax);

% Store for next iteration
data.u_prev = u_scaled;

% Descale to physical units and clip
u_cmd = u_scaled * data.u_std + data.u_mean;
u_cmd = min(max(u_cmd, data.u_min_phys), data.u_max_phys);

end

function data = initialise_controller()

data = struct();

% Strejc model parameters (discrete-time, Ts = 1 s)
data.A = 0.98540172;
data.B = 0.01640409;
data.C = 1;
data.nx = 1;

% Scaling factors
scaling = load_scaling();
data.x_mean = scaling.x_mean;
data.x_std = scaling.x_std;
data.u_mean = scaling.u_mean;
data.u_std = scaling.u_std;

% Kalman filter parameters (copied from cl_strejc.m)
data.Q_KF = 0.5;
data.R_KF = 0.1;
data.P0 = 1;

% Physical limits
data.u_min_phys = 20;
data.u_max_phys = 100;
data.umin = (data.u_min_phys - data.u_mean) / data.u_std;
data.umax = (data.u_max_phys - data.u_mean) / data.u_std;

ymin_phys = 0;
ymax_phys = 70;
data.ymin = (ymin_phys - data.x_mean) / data.x_std;
data.ymax = (ymax_phys - data.x_mean) / data.x_std;

% MPC weights and horizon
Qy = 10;
Qu = 1;
N = 40;

% Build optimiser
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

data.controller = optimizer(constraints, objective, ...
    sdpsettings('solver','quadprog','verbose',0), ...
    [x0_param; r_param], u{1});

% Persistent states
data.initialised = false;
data.x_est = 0;
data.P = data.P0;
data.u_prev = 0;
data.u0 = 0;

end

function scaling = load_scaling()
scaling = struct();

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

