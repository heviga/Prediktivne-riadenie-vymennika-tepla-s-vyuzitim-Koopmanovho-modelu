%% Comparison of Koopman vs Strejc Open-Loop Predictions
% This script compares the open-loop predictions from both models
clear all, close all

%% Load data from both identification scripts
% Load test data first
load('test_data.mat');  % Ytest, Utest
Ytest = Ytest(:);
Utest = Utest(:);

% Load Strejc results
if exist('strejc_ol_data.mat', 'file')
    load('strejc_ol_data.mat');
    y_strejc = y_open_desc;
    time_strejc = time;
    u_strejc = u_open_desc;
    fprintf('✓ Loaded Strejc data\n');
else
    fprintf('❌ strejc_ol_data.mat not found - run ident_strejc.m first\n');
    return;
end

% Load Koopman results  
if exist('koopman_ol.mat', 'file')
    load('koopman_ol.mat');
    y_koopman = y_koopman_desc;
    time_koopman = time;
    u_koopman = Utest;
    fprintf('✓ Loaded Koopman data\n');
else
    fprintf('❌ koopman_ol.mat not found - run ident_koopman.m first\n');
    return;
end

%% Ensure same time vector
% Check lengths and use the shortest one
fprintf('Data lengths:\n');
fprintf('  Ytest: %d\n', length(Ytest));
fprintf('  y_strejc: %d\n', length(y_strejc));
fprintf('  y_koopman: %d\n', length(y_koopman));

% Use the shortest length to avoid dimension mismatch
min_length = min([length(Ytest), length(y_strejc), length(y_koopman)]);
fprintf('Using min_length: %d\n', min_length);

time_common = 0:(min_length-1);

y_strejc = y_strejc(1:min_length);
y_koopman = y_koopman(1:min_length);
y_true = Ytest(1:min_length);
u_common = u_koopman(1:min_length);

%% Calculate performance metrics
% RMSE
rmse_strejc = sqrt(mean((y_strejc - y_true).^2));
rmse_koopman = sqrt(mean((y_koopman - y_true).^2));

% MAE
mae_strejc = mean(abs(y_strejc - y_true));
mae_koopman = mean(abs(y_koopman - y_true));

% R²
ss_res_strejc = sum((y_true - y_strejc).^2);
ss_tot_strejc = sum((y_true - mean(y_true)).^2);
r2_strejc = 1 - (ss_res_strejc / ss_tot_strejc);

ss_res_koopman = sum((y_true - y_koopman).^2);
ss_tot_koopman = sum((y_true - mean(y_true)).^2);
r2_koopman = 1 - (ss_res_koopman / ss_tot_koopman);

%% Display results
fprintf('\n=== PERFORMANCE COMPARISON ===\n');
fprintf('Model        | RMSE    | MAE     | R²      \n');
fprintf('-------------|---------|---------|---------\n');
fprintf('Strejc       | %.4f  | %.4f  | %.4f  \n', rmse_strejc, mae_strejc, r2_strejc);
fprintf('Koopman      | %.4f  | %.4f  | %.4f  \n', rmse_koopman, mae_koopman, r2_koopman);
fprintf('-------------|---------|---------|---------\n');

if rmse_koopman < rmse_strejc
    fprintf('🏆 Koopman model performs better (lower RMSE)\n');
else
    fprintf('🏆 Strejc model performs better (lower RMSE)\n');
end

%% Create comparison plots
figure();

% Main comparison plot
subplot(2,1,1);
plot(time_common, y_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True Output'); hold on;
plot(time_common, y_strejc, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Strejc');
plot(time_common, y_koopman, 'r:', 'LineWidth', 1.5, 'DisplayName', 'Koopman');
xlabel('Time step');
ylabel('Output y (°C)');
title('Model Comparison: Open-Loop Predictions');
legend('Location', 'best');
grid on;
% 
% % Error comparison
% subplot(2,2,2);
% error_strejc = y_strejc - y_true;
% error_koopman = y_koopman - y_true;
% plot(time_common, error_strejc, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Strejc Error'); hold on;
% plot(time_common, error_koopman, 'r:', 'LineWidth', 1.5, 'DisplayName', 'Koopman Error');
% xlabel('Time step');
% ylabel('Prediction Error (°C)');
% title('Prediction Errors');
% legend('Location', 'best');
% grid on;

% Input signal
subplot(2,1,2);
plot(time_common, u_common, 'g-', 'LineWidth', 1.5);
xlabel('Time step');
ylabel('Input u');
title('Input Signal (Test Data)');
grid on;

% Performance metrics bar chart
% subplot(2,2,4);
% metrics = [rmse_strejc, rmse_koopman; mae_strejc, mae_koopman; r2_strejc, r2_koopman];
% bar(metrics);
% set(gca, 'XTickLabel', {'RMSE', 'MAE', 'R²'});
% ylabel('Value');
% title('Performance Metrics');
% legend('Strejc', 'Koopman', 'Location', 'best');
% grid on;

%% Save comparison results
save('ident_comparison_results.mat', ...
    'y_strejc', 'y_koopman', 'y_true', 'time_common', 'u_common', ...
    'rmse_strejc', 'rmse_koopman', 'mae_strejc', 'mae_koopman', ...
    'r2_strejc', 'r2_koopman');

fprintf('\n✅ Comparison complete! Results saved to comparison_results.mat\n');
fprintf('📊 Plots displayed - check the figure window\n');