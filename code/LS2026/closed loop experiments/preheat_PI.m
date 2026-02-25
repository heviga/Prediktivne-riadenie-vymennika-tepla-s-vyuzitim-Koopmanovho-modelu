function [y_T4, y_T2, u_Pump2] = preheat_PI(pct23, setpoint_T4, Ts, P_spiral, Pump1_const, hold_loops)
% PREHEAT_PI - PI control of Pump2 + Heater to reach a target T4
% Stops automatically when T4 reaches setpoint and holds for `hold_loops` cycles
%
% Inputs:
%   pct23        - ELab device object
%   setpoint_T4  - target temperature (°C)
%   Ts           - sampling time (s)
%   P_spiral     - P-gain for Heater
%   Pump1_const  - constant Pump1 setting
%   hold_loops   - number of consecutive loops T4 must stay at setpoint ±tolerance
%
% Outputs:
%   y_T4         - measured T4 over time
%   y_T2         - measured T2 over time
%   u_Pump2      - Pump2 command over time

    tic_total = tic;  % start overall timer

    % PI controller parameters (jemnejšie pre stabilnú reguláciu)
    Pump2_base = 50;
    Pump2_min = 0;
    Pump2_max = 100;
    Kp_pump2 = 6.0;
    Ki_pump2 = 0.8;
    integral_error = 0;
    tolerance = 0.15;  % ±0.1 °C okolo setpointu

    % Preallocate logs (max 20 min)
    max_steps = 20*60/Ts;
    y_T4 = zeros(max_steps,1);
    y_T2 = zeros(max_steps,1);
    u_Pump2 = zeros(max_steps,1);

    % Quick read to ensure tags are available
    double(pct23.getTag('T4').value); pause(1);
    double(pct23.getTag('T2').value); pause(1);

    hold_count = 0;
    k = 0;

    while hold_count < hold_loops && k < max_steps
        k = k + 1;
        tic;

        % Measurements
        T4 = double(pct23.getTag('T4').value);
        T2 = double(pct23.getTag('T2').value);
        y_T4(k) = T4;
        y_T2(k) = T2;

        % Heater P-control
        value_sp = min(max(P_spiral * (76 - T2), 0), 100);

        % Pump2 PI-control
        error = setpoint_T4 - T4;
        integral_error = integral_error + error*Ts;

        % Anti-windup: reset integral if output saturates
        u_raw = Pump2_base + Kp_pump2*error + Ki_pump2*integral_error;
        if u_raw > Pump2_max
            u_Pump2(k) = Pump2_max;
            integral_error = integral_error - error*Ts; % undo last step
        elseif u_raw < Pump2_min
            u_Pump2(k) = Pump2_min;
            integral_error = integral_error - error*Ts;
        else
            u_Pump2(k) = u_raw;
        end

        % Apply control
        pct23.setTag('Pump2', u_Pump2(k));
        pct23.setTag('Pump1', Pump1_const);
        pct23.setTag('Heater', value_sp);
        pct23.setTag('FSV', 1);

        % Display info
        fprintf('loop: %d, T4 = %.3f °C, Pump2 = %.3f %%, Heater = %.3f %%\n', k, T4, u_Pump2(k), value_sp);

        % Check if T4 is within tolerance of setpoint
        if abs(T4 - setpoint_T4) <= tolerance
            hold_count = hold_count + 1;
        else
            hold_count = 0;
        end

        % Maintain sampling time
        pause(max(0, Ts - toc));
    end

    % Trim arrays to actual size
    y_T4 = y_T4(1:k);
    y_T2 = y_T2(1:k);
    u_Pump2 = u_Pump2(1:k);

    elapsed_total = toc(tic_total);
    fprintf('=== Preheat PI function completed in %.2f seconds ===\n', elapsed_total);
end
