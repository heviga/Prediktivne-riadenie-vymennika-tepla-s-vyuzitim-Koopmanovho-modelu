% Steady-state gain K ≈ 1.1237
% Time constant (tau) ≈ 68 samples

%% Initialization
clc; close all; clear;

% ===== LaTeX plot settings =====
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultLegendFontName','Times New Roman');

label_fs = 14;
title_fs = 16;
tick_fs  = 12;
line_w   = 1.3;

% ===== Output folder =====
fig_dir = 'figs';
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

%% Load data
data = load('identifikacia2.mat'); % 3251 samples

% Temperature T4
y = data.Temperatures{4}.Values.Data;

% Time
t = data.tout;

% Input: Pump F / hot circuit pump
u = data.uout(:,2);

% Remove first bad part of the measurement
u = u(251:end);
y = y(251:end);
t = t(251:end);

% Reset time to start from 1
t = t - t(1) + 1;

%% Basic statistics
y_mean = mean(y); % 58.3377
u_mean = mean(u); % 63.0883
y_std  = std(y);  % 7.1204
u_std  = std(u);  % 23.9910

fprintf('y_mean = %.4f\n', y_mean);
fprintf('u_mean = %.4f\n', u_mean);
fprintf('y_std  = %.4f\n', y_std);
fprintf('u_std  = %.4f\n', u_std);

%% Raw measurement overview
fig0 = figure('Color','w','Name','Raw Measurements');

subplot(2,1,1);
plot(t, u, 'LineWidth', line_w);
grid on; box on; grid minor;
ylabel('Pump F speed ($\%$)', 'FontSize', label_fs);
title('Input signal', 'FontSize', title_fs);
set(gca,'FontSize',tick_fs);

subplot(2,1,2);
plot(t, y, 'LineWidth', line_w);
grid on; box on; grid minor;
xlabel('Time (s)', 'FontSize', label_fs);
ylabel('Outlet temperature $T_4$ ($^\circ$C)', 'FontSize', label_fs);
title('Measured output', 'FontSize', title_fs);
set(gca,'FontSize',tick_fs);

exportgraphics(fig0, fullfile(fig_dir,'fig_raw_measurements.png'), 'Resolution', 300);

%% Step detection
step_indices = find(abs(diff(u)) > 1);
num_steps = numel(step_indices);

fprintf('Detected steps: %d\n', num_steps);

%% Signal normalization
y_scaled = (y - y_mean) / y_std;
u_scaled = (u - u_mean) / u_std;

delay = 10;          % samples after step for input value
max_length = 250;    % step response length

y_steps = NaN(num_steps, max_length);
u_steps = NaN(num_steps, 1);

%% Plot all normalized step responses
colors = lines(max(num_steps, 1));

fig1 = figure('Color','w','Name','Normalized Step Responses');
hold on; grid on; box on; grid minor;

title('Normalized step responses', 'FontSize', title_fs);
xlabel('Time since step (s)', 'FontSize', label_fs);
ylabel('Normalized output', 'FontSize', label_fs);
set(gca,'FontSize',tick_fs);

for i = 1:num_steps
    idx = step_indices(i);

    % Bounds check
    if idx - 1 < 1 || idx + delay >= length(u) || idx + max_length - 1 > length(y)
        fprintf('Skipping step %d: not enough samples.\n', i);
        continue
    end

    % Input change in scaled coordinates
    u_before = u_scaled(idx - 1);
    u_after  = u_scaled(idx + delay);
    delta_u  = u_after - u_before;

    if abs(delta_u) < eps
        fprintf('Skipping step %d: zero input change.\n', i);
        continue
    end

    % Output segment
    y_step_scaled = y_scaled(idx : idx + max_length - 1);

    % Normalized response to unit positive step
    y_norm = (y_step_scaled - y_scaled(idx - 1)) / abs(delta_u);

    if delta_u < 0
        y_norm = -y_norm;
    end

    % Store and plot
    y_steps(i,:) = y_norm(:)';
    u_steps(i)   = delta_u;

    plot(0:max_length-1, y_norm, ...
        'Color', colors(i,:), ...
        'LineWidth', 1.0);

    fprintf('Step %d: delta u scaled = %.4f\n', i, delta_u);
end

hold off;

exportgraphics(fig1, fullfile(fig_dir,'fig_normalized_step_responses.png'), 'Resolution', 300);
exportgraphics(fig1, fullfile(fig_dir,'fig_normalized_step_responses.pdf'), 'ContentType','vector');

%% Average normalized step response
valid_rows = ~all(isnan(y_steps), 2);
valid_steps = sum(valid_rows);

fprintf('Valid step responses: %d / %d\n', valid_steps, num_steps);

avg_step = mean(y_steps, 1, 'omitnan');

fig2 = figure('Color','w','Name','Average Normalized Step Response');
hAvg = plot(0:max_length-1, avg_step, 'k', 'LineWidth', 2.2);
hold on; grid on; box on; grid minor;

title('Average normalized step response', 'FontSize', title_fs);
xlabel('Time since step (s)', 'FontSize', label_fs);
ylabel('Normalized output', 'FontSize', label_fs);
set(gca,'FontSize',tick_fs);

%% Strejc model parameter estimation
K = mean(avg_step(end-6:end), 'omitnan');
target_value = 0.632 * K;
tau_idx = find(avg_step >= target_value, 1, 'first');

fprintf('\nSteady-state gain K ≈ %.4f\n', K);

if ~isempty(tau_idx)
    fprintf('Time constant tau ≈ %d samples\n', tau_idx);
else
    warning('Tau could not be determined: target value was not reached.');
end

% Horizontal line: 63.2 % of K
hY = yline(target_value, '--', '$63.2\%$ of $K$', ...
    'Interpreter','latex', ...
    'LineWidth', 1.1, ...
    'LabelHorizontalAlignment','right', ...
    'LabelVerticalAlignment','bottom');

% Vertical line: time constant
if ~isempty(tau_idx)
    hX = xline(tau_idx, '--r', '$T$', ...
        'Interpreter','latex', ...
        'LineWidth', 1.1, ...
        'LabelOrientation','horizontal', ...
        'LabelVerticalAlignment','top');

    legend([hAvg hY hX], ...
        {'Average response', '$63.2\%$ of $K$', '$T$'}, ...
        'Interpreter','latex', ...
        'Location','best', ...
        'FontSize', tick_fs);
else
    legend([hAvg hY], ...
        {'Average response', '$63.2\%$ of $K$'}, ...
        'Interpreter','latex', ...
        'Location','best', ...
        'FontSize', tick_fs);
end

hold off;

exportgraphics(fig2, fullfile(fig_dir,'fig_average_step_response.png'), 'Resolution', 300);
exportgraphics(fig2, fullfile(fig_dir,'fig_average_step_response.pdf'), 'ContentType','vector');

%% Saving train and test data
Ytrain = y(1:2000);
Utrain = u(1:2000);

Ytest = y(2001:end);
Utest = u(2001:end);

% save("train_data.mat", 'Ytrain', 'Utrain')
% save("test_data.mat", 'Ytest', 'Utest')

%% Raw measured signals used for identification with train/test split
fig3 = figure('Color','w','Name','Raw Identification Measurements');

% Colors
c_light = [0.6 0.8 1.0];   % light blue
c_dark  = [0 0.2 0.6];     % dark blue

idx_train = 1:2000;
idx_test  = 2001:length(t);

subplot(2,1,1); hold on;
plot(t(idx_train), u(idx_train), ...
    'Color', c_light, ...
    'LineWidth', line_w);

plot(t(idx_test), u(idx_test), ...
    'Color', c_dark, ...
    'LineWidth', line_w);

grid on; box on; grid minor;
ylabel('Pump F speed ($\%$)', 'FontSize', label_fs);
title('Input signal', 'FontSize', title_fs);
legend('Training data','Test data', ...
    'Location','southwest', ...
    'Interpreter','latex', ...
    'FontSize', tick_fs);
ylim([0 101]);
xlim([0 3000]);
set(gca,'FontSize',tick_fs);
hold off;

subplot(2,1,2); hold on;
plot(t(idx_train), y(idx_train), ...
    'Color', c_light, ...
    'LineWidth', line_w);

plot(t(idx_test), y(idx_test), ...
    'Color', c_dark, ...
    'LineWidth', line_w);

grid on; box on; grid minor;
xlabel('Time (s)', 'FontSize', label_fs);
ylabel('Outlet temperature $T_4$ ($^\circ$C)', 'FontSize', label_fs);
title('Measured output', 'FontSize', title_fs);
legend('Training data','Test data', ...
    'Location','best', ...
    'Interpreter','latex', ...
    'FontSize', tick_fs);
xlim([0 3000]);
set(gca,'FontSize',tick_fs);
hold off;

exportgraphics(fig3, fullfile(fig_dir,'fig_raw_measurements_split.png'), 'Resolution', 300);
exportgraphics(fig3, fullfile(fig_dir,'fig_raw_measurements_split.pdf'), 'ContentType','vector');