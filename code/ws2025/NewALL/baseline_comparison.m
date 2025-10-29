% Baseline Comparison: Koopman vs Strejc MPC
% Simple comparison using saved results from both model files
%ofsst na kalmanovi???

clc, clear all, %close all

% Add path to readNPY function
addpath('../');
% Load data
load('train_data.mat');  % Ytrain, Utrain
load('test_data.mat');   % Ytest, Utest

% Flatten
Ytrain = Ytrain(:);
Utrain = Utrain(:);
Ytest = Ytest(:);
Utest = Utest(:);

% Full dataset for scaling
Yall = [Ytrain; Ytest];
Uall = [Utrain; Utest];

x_mean = mean(Yall);
x_std = std(Yall);
u_mean = mean(Uall);
u_std = std(Uall);

%% LOAD RESULTS FROM BOTH MODELS

% Load Koopman results with baseline
fprintf('Loading Koopman results...\n');
load('results_koopman_to_zero.mat');
load('baseline_reference.mat');  % Koopman baseline
koop_time = 0:length(y_true_desc)-1;
koop_y_true_desc = y_true_desc;  % Koopman baseline
koop_y_est_desc = y_est_desc;    % Koopman KF estimate
koop_u_cl_desc = u_cl_desc;
y_true_koop = y_true;
u_koop = u_cl;
% Load Strejc results with baseline
fprintf('Loading Strejc results...\n');
load('results_strejc_to_zero.mat');
load('baseline_reference_strejc.mat');  % Strejc baseline
strejc_time = 0:length(y_true_desc)-1;
strejc_y_true_desc = y_true_desc;  % Strejc baseline
strejc_y_est_desc = y_est_desc;    % Strejc KF estimate  
strejc_u_cl_desc = u_cl_desc;
y_true_strejc = y_true;
u_strejc = u_cl;

%% COMPARISON PLOT
figure();

% First subplot: Koopman MPC + Kalman Filter vs Baseline
subplot(2,1,1)
plot(koop_time, koop_y_true_desc, 'm-', 'LineWidth', 1.5, 'DisplayName', 'True output (Baseline)'); hold on;
plot(koop_time, koop_y_est_desc, 'b--', 'LineWidth', 2.5, 'DisplayName', 'KF estimate (Koopman)');
xlabel('Time step'); ylabel('Output y (°C)');
yline(x_mean)
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
yline(x_mean)
title('Strejc MPC + Kalman Filter vs Baseline System');
grid on; grid minor;
ylim([50 68]);

figure();

% Jediný graf: Koopman MPC + KF a Strejc MPC + KF vs Baseline
plot(koop_time, koop_y_true_desc, 'm-', 'LineWidth', 1.5, 'DisplayName', 'True output (Baseline)'); hold on;
plot(koop_time, koop_y_est_desc, 'b--', 'LineWidth', 2.5, 'DisplayName', 'KF estimate (Koopman)');
plot(strejc_time, strejc_y_est_desc, 'r--', 'LineWidth', 2.5, 'DisplayName', 'KF estimate (Strejc)');

% Osi, legenda a popisy
xlabel('Time step');
ylabel('Output y (°C)');
legend('Location', 'best');
yline(x_mean, 'k:', 'DisplayName', 'Mean temperature');
title('Koopman vs Strejc MPC + Kalman Filter (Baseline Comparison)');
grid on; grid minor;
ylim([50 60]);

%%
Qy = 10;
Qu = 1;
J_koopY=0;
J_koopU=0;
J_strejcU = 0;
J_strejcY = 0;

for i =1:length(strejc_time)-1
    J_koopY = J_koopY + y_true_koop(i)' * Qy * y_true_koop(i);
    J_koopU=J_koopU+ u_koop(i)' * Qu*u_koop(i);
    J_strejcU = J_strejcU+ u_strejc(i)' * Qu*u_strejc(i);
    J_strejcY = J_strejcY+ y_true_strejc(i)' * Qy*y_true_strejc(i);
end

fprintf('Koopman Model Cost (Output): J_koopY = %.4f\n', J_koopY);
fprintf('Koopman Model Cost (Control): J_koopU = %.4f\n', J_koopU);
fprintf('Strejc Model Cost (Output): J_strejcY = %.4f\n', J_strejcY);
fprintf('Strejc Model Cost (Control): J_strejcU = %.4f\n', J_strejcU);
fprintf('Koopman Model Cost (sum): J_koopY + U = %.4f\n', (J_koopY+J_koopU));
fprintf('Strejc Model Cost (sum): J_koopY + U = %.4f\n', (J_strejcY+J_strejcU));
fprintf('Koopman Model Cost (sum ratio to Strejc): (J_koopY + J_koopU) / (J_strejcY + J_strejcU) = %.4f\n', ...
    (J_koopY + J_koopU) / (J_strejcY + J_strejcU));
%% CALCULATE PERFORMANCE METRICS
% Calculate RMSE for both models (using their respective baselines)
koop_rmse = sqrt(mean((koop_y_true_desc - x_mean).^2));
strejc_rmse = sqrt(mean((strejc_y_true_desc - x_mean).^2));

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