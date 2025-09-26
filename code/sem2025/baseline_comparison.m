% Baseline Comparison: Koopman vs Strejc MPC
% Simple comparison using saved results from both model files

clc, clear all, %close all

% Add path to readNPY function
addpath('../');

%% LOAD RESULTS FROM BOTH MODELS

% Load Koopman results with baseline
fprintf('Loading Koopman results...\n');
load('results_koopman.mat');
load('baseline_reference.mat');  % Koopman baseline
koop_time = 0:length(y_true_desc)-1;
koop_y_true_desc = y_true_desc;  % Koopman baseline
koop_y_est_desc = y_est_desc;    % Koopman KF estimate
koop_u_cl_desc = u_cl_desc;

% Load Strejc results with baseline
fprintf('Loading Strejc results...\n');
load('results_strejc_to_zero.mat');
load('baseline_reference_strejc.mat');  % Strejc baseline
strejc_time = 0:length(y_true_desc)-1;
strejc_y_true_desc = y_true_desc;  % Strejc baseline
strejc_y_est_desc = y_est_desc;    % Strejc KF estimate  
strejc_u_cl_desc = u_cl_desc;

%% COMPARISON PLOT
figure();

% First subplot: Koopman MPC + Kalman Filter vs Baseline
subplot(2,1,1)
plot(koop_time, koop_y_true_desc, 'm-', 'LineWidth', 1.5, 'DisplayName', 'True output (Baseline)'); hold on;
plot(koop_time, koop_y_est_desc, 'b--', 'LineWidth', 2.5, 'DisplayName', 'KF estimate (Koopman)');
xlabel('Time step'); ylabel('Output y (°C)');
legend('Location', 'best');
title('Koopman MPC + Kalman Filter vs Baseline System');
grid on; grid minor;
ylim([50 68]);

% Second subplot: Strejc MPC + Kalman Filter vs Baseline
subplot(2,1,2)
plot(strejc_time, strejc_y_true_desc, 'm-', 'LineWidth', 1.5, 'DisplayName', 'True output (Baseline)'); hold on;
plot(strejc_time, strejc_y_est_desc, 'r--', 'LineWidth', 2.5, 'DisplayName', 'KF estimate (Strejc)');
xlabel('Time step'); ylabel('Output y (°C)');
legend('Location', 'best');
title('Strejc MPC + Kalman Filter vs Baseline System');
grid on; grid minor;
ylim([50 68]);

%% CALCULATE PERFORMANCE METRICS
% Calculate RMSE for both models (using their respective baselines)
koop_rmse = sqrt(mean((koop_y_true_desc - koop_y_est_desc).^2));
strejc_rmse = sqrt(mean((strejc_y_true_desc - strejc_y_est_desc).^2));

fprintf('\n=== PERFORMANCE COMPARISON ===\n');
fprintf('Koopman RMSE: %.4f °C\n', koop_rmse);
fprintf('Strejc RMSE:  %.4f °C\n', strejc_rmse);
if strejc_rmse > 0
    fprintf('Improvement: %.2f%%\n', (strejc_rmse - koop_rmse)/strejc_rmse * 100);
end

%% SAVE COMPARISON RESULTS
% save('baseline_comparison_results.mat', 'koop_time', 'koop_y_true_desc', 'koop_y_est_desc', 'koop_u_cl_desc', ...
%      'strejc_time', 'strejc_y_true_desc', 'strejc_y_est_desc', 'strejc_u_cl_desc', ...
%      'koop_rmse', 'strejc_rmse');
% 
% fprintf('\nResults saved to baseline_comparison_results.mat\n');