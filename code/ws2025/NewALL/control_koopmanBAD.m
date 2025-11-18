function u_cmd = control_koopman(y_meas, r_setpoint, opts)
%CONTROL_KOOPMAN Koopman-based MPC with Kalman filtering for pct23 temperature control.
%
%   u_cmd = CONTROL_KOOPMAN(y_meas, r_setpoint) returns the Pump2 command
%   given the latest temperature measurement y_meas (C) and the desired
%   setpoint r_setpoint (C).
%
%   u_cmd = CONTROL_KOOPMAN(y_meas, r_setpoint, opts) allows additional
%   options. Supported field:
    %       - opts.reset : when true, reinitialises internal persistent state.
    
persistent ctrl

if nargin < 2 || isempty(r_setpoint)
    r_setpoint = 60;
    end
    
if nargin < 3
    opts = struct();
    end
    
if isfield(opts, 'reset') && opts.reset
    ctrl = [];
        u_cmd = NaN;
            return;
            end
            
if isempty(ctrl)
    ctrl = koopman_init();
    end
    
if isempty(y_meas)
    error('control_koopman:MissingMeasurement', ...
            'Measurement y_meas must be provided unless opts.reset is true.');
            end
            
% Scale measurement and reference
y_scaled = (y_meas - ctrl.x_mean) / ctrl.x_std;
r_scaled = (r_setpoint - ctrl.x_mean) / ctrl.x_std;

% Kalman filter state update
ctrl = koop_kalman(ctrl, y_scaled);

% MPC optimisation in scaled space
params = [ctrl.x_est; r_scaled];
u_scaled = ctrl.controller{params};
u_scaled = full(u_scaled);
u_scaled = min(max(u_scaled, ctrl.umin), ctrl.umax);

% Store for next iteration
ctrl.u_prev = u_scaled;

% Convert to physical units and apply final safety clip
u_cmd = u_scaled * ctrl.u_std + ctrl.u_mean;
u_cmd = min(max(u_cmd, ctrl.u_min_phys), ctrl.u_max_phys);

end
