clc; clear; close all;
%%
% Load results
load('results_koopman.mat', 'y_cl_desc', 'u_cl_desc');
y_koop = y_cl_desc(:);
u_koop = u_cl_desc(:);

load('results_strejc_to_zero.mat', 'y_cl_desc', 'u_cl_desc');
y_strejc = y_cl_desc(:);
u_strejc = u_cl_desc(:);

sim_length = length(y_koop) - 1;
time = 0:sim_length;

% Control effort (sum of absolute values)
effort_koop = sum(abs(u_koop));
effort_strejc = sum(abs(u_strejc));

% Output magnitude
y_sum_koop = sum(abs(y_koop));
y_sum_strejc = sum(abs(y_strejc));

% Optional: error to 0°C
e_koop = y_koop;  % reference = 0
e_strejc = y_strejc;

e_sum_koop = sum(abs(e_koop));
e_sum_strejc = sum(abs(e_strejc));

rmse_koop = sqrt(mean((y_koop).^2));
rmse_strejc = sqrt(mean((y_strejc).^2));

% Print comparison
fprintf('\n--- MPC Comparison ---\n');
fprintf('Sum |u|     Koopman: %.2f \t Strejc: %.2f\n', effort_koop, effort_strejc);
fprintf('Sum |y|     Koopman: %.2f \t Strejc: %.2f\n', y_sum_koop, y_sum_strejc);
fprintf('Sum |e|     Koopman: %.2f \t Strejc: %.2f\n', e_sum_koop, e_sum_strejc);

fprintf('RMSE (MPC Strejc vs 60 °C)   = %.4f\n', rmse_strejc);
fprintf('RMSE (MPC Koopman vs 60 °C)  = %.4f\n', rmse_koop);


figure;
subplot(2,1,1)
plot(time, y_strejc , 'b-', 'LineWidth', 2); hold on;
plot(time, y_koop, 'm--', 'LineWidth', 2);
xlabel('Time step'); ylabel('Output y (°C)');
legend('MPC Strejc', 'MPC Koopman');
title('Closed-loop Output: Strejc vs Koopman');
grid on;
ylim([45 65])

subplot(2,1,2)
stairs(time(1:end-1), u_strejc, 'b-', 'LineWidth', 2); hold on;
stairs(time(1:end-1), u_koop, 'm--', 'LineWidth', 2);
xlabel('Time step'); ylabel('Input u');
legend('MPC Strejc', 'MPC Koopman');
title('Control Input Comparison');
grid on;


%% --- Open-loop Identification Comparison ---
% Load Strejc data
load('strejc_open_loop_comparison_data.mat', ...
    'y_open_desc', 'y_true', 'time', 'u_open_desc');

y_true = y_true(:);
y_open_strejc = y_open_desc(:);


% Load Koopman data
load('koopman_open_loop_comparison.mat', ...
    'y_koopman_desc');

y_open_koopman = y_koopman_desc(:);

y_open_strejc = y_open_strejc(2:end);
y_open_koopman = y_open_koopman(2:end);
y_open_desc = y_open_desc(1:end-1);
time = time(1:end-1);

% Compute RMSE (against the same y_true)
rmse_open_strejc = sqrt(mean((y_open_strejc - y_true).^2));
rmse_open_koopman = sqrt(mean((y_open_koopman - y_true).^2));

% Display RMSE
fprintf('\n--- Open-loop Prediction RMSE ---\n');
fprintf('RMSE (Open-loop Strejc)   = %.4f °C\n', rmse_open_strejc);
fprintf('RMSE (Open-loop Koopman)  = %.4f °C\n', rmse_open_koopman);

% Plot Open-loop Output Comparison
figure;
subplot(2,1,1)
plot(time, y_true, 'k:', 'LineWidth', 1.5); hold on;
plot(time, y_open_strejc, 'b-', 'LineWidth', 2);
plot(time, y_open_koopman, 'm--', 'LineWidth', 2);
xlabel('Time step'); ylabel('Output y (°C)');
legend('True Output', 'Strejc Prediction', 'Koopman Prediction');
title('Open-loop Output Comparison');
grid on;

subplot(2,1,2)
stairs(time, u_open_desc, 'b', 'LineWidth', 2); hold on;
xlabel('Time step'); ylabel('Input u');
legend('True Input');
title('Open-loop Input Comparison');
grid on;
