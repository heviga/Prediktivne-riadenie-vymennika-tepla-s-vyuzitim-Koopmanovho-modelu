clc; clear; close all

%% ----------------------------
% LOAD DATA (open-loop predictions)
% ----------------------------

% Load Strejc open loop
load('strejc_open_loop_comparison_data.mat', 'y_open_desc', 'y_true', 'u_open_desc', 'time');
y_open_strejc = y_open_desc(:);
u_open_strejc = u_open_desc(:);
y_true_open = y_true(:);

% Load Koopman open loop
load('koopman_open_loop_comparison.mat', 'y_koopman_desc');
y_open_koopman = y_koopman_desc(:);

%% ----------------------------
% METRICS CALCULATION
% ----------------------------

y_open_strejc = y_open_strejc(2:end);
y_open_koopman = y_open_koopman(2:end);

% RMSE Open loop
rmse_open_strejc = sqrt(mean((y_open_strejc - y_true_open).^2));
rmse_open_koopman = sqrt(mean((y_open_koopman - y_true_open).^2));

% ISE Open loop
ise_open_strejc = sum((y_open_strejc - y_true_open).^2);
ise_open_koopman = sum((y_open_koopman - y_true_open).^2);

%% ----------------------------
% PLOTS
% ----------------------------

figure('Name','Open-loop Identification','NumberTitle','off');
subplot(2,1,1)
plot(y_true_open, 'k:', 'LineWidth', 1.5); hold on;
plot(y_open_strejc, 'b-', 'LineWidth', 2);
plot(y_open_koopman, 'm--', 'LineWidth', 2);
legend('True','Strejc','Koopman');
xlabel('Time step'); ylabel('Output y (°C)');
title(sprintf('Open-loop Output Comparison (RMSE Strejc = %.2f °C, Koopman = %.2f °C)', rmse_open_strejc, rmse_open_koopman));
grid on;

subplot(2,1,2)
stairs(u_open_strejc, 'LineWidth', 2);
xlabel('Time step'); ylabel('Input u');
title('Open-loop Input during Test');
grid on;

%% ----------------------------
% SUMMARY
% ----------------------------
fprintf('\n==== OPEN LOOP IDENTIFICATION RESULTS ====\n');
fprintf('RMSE Strejc:   %.2f °C\n', rmse_open_strejc);
fprintf('RMSE Koopman:  %.2f °C\n', rmse_open_koopman);
fprintf('ISE Strejc:    %.2f\n', ise_open_strejc);
fprintf('ISE Koopman:   %.2f\n', ise_open_koopman);
fprintf('===========================================\n');
