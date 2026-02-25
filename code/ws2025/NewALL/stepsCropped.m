%% load data
clc,close all
data = load('identifikacia2.mat'); %3251 samples

%teplota
y = data.Temperatures{4}.Values.Data;% temperatures t4
%cas
t = data.tout;
%input
u=data.uout(:,2);

u=u(251:end);
y=y(251:end);
t=t(251:end);

% reset time to start from 1
t = t - t(1) + 1;


% disp(size(y))
% disp(size(t))
% disp(size(u))
figure
subplot(2,1,1)
plot(t,u)
grid on, box on, grid minor
subplot(2,1,2)
plot(t,y)
grid on, box on, grid minor
title('ciste merania');

y_mean = mean(y); % 58.3377
u_mean = mean(u); % 63.0883
y_std = std(y); % 7.1204
u_std = std(u); % 23.9910


%% hladanie stepov
step_indices = find(abs(diff(u)) >1);

num_steps = length(step_indices);

y_scaled = (y - y_mean) / y_std;
u_scaled = (u - u_mean) / u_std;

delay = 10;
pre_window = 1;
max_length = 250;

y_steps = NaN(num_steps, max_length);

u_steps = NaN(num_steps, 1);
colors = lines(num_steps);


%% Plot normalized and true (unscaled) step responses side by side
figure; hold on;
title('Normalized Step Responses');
xlabel('Time (s)');
ylabel('Normalized Output');
grid on; grid minor;box on;

for i = 1:num_steps
    start_idx = step_indices(i)

    % Bounds check
    if start_idx - 1 < 1 || (start_idx + delay >= length(u))
        fprintf('Skipping step %d: not enough delay margin.\n', i);
        continue
    end

    % Δu and normalization
    u_before = u_scaled(start_idx - 1);
    u_after  = u_scaled(start_idx + delay);
    delta_u = u_after - u_before;

    y_step_scaled = y_scaled(start_idx : start_idx + 249);

    y_norm = (y_step_scaled - y_scaled(start_idx - 1)) / abs(delta_u);
    if delta_u < 0
        y_norm = -y_norm;
    end

    % Store and plot
    y_steps(i, :) = y_norm(:)';
    u_steps(i) = delta_u;

    plot(0:max_length - 1, y_norm, 'Color', colors(i,:), 'LineWidth', 1.2);

    fprintf('Step %d: Δu = %.1f\n', i, delta_u);
end

% Save figure
%saveas(gcf, 'C:\Users\ivadu\Desktop\8.semestrik\vymennik\prez\all_step_scaled_only.png');

valid_rows = ~any(isnan(y_steps), 2);
avg_step = mean(y_steps(valid_rows, :), 1);

valid_steps = sum(~isnan(y_steps(:,1)));
fprintf('Valid step responses: %d / %d\n', valid_steps, num_steps);

avg_step = mean(y_steps, 1, 'omitnan');

figure;
hAvg = plot(0:max_length-1, avg_step, 'k', 'LineWidth', 2);
title('Average Normalized Step Response');
xlabel('Time (s)'); ylabel('Normalized Output');
grid on;box on;grid minor;

% Estimate gain and time constant
K = mean(avg_step(end-6:end), 'omitnan'); % 1.1237

target_value = 0.632 * K;
tau_idx = find(avg_step >= target_value, 1, 'first'); % 68

fprintf('\nSteady-state gain K ≈ %.4f\n', K);
if ~isempty(tau_idx)
    fprintf('Time constant (tau) ≈ %d samples\n', tau_idx);
else
    warning('Tau could not be determined: target value not reached.');
end

% Annotate plot
hold on;
hY = yline(target_value, '--', '$63.2\%$ of $\mathrm{K}$', 'Interpreter', 'latex');
if ~isempty(tau_idx)
    hX = xline(tau_idx, '--r', '$\mathrm{T}$', 'Interpreter', 'latex', 'LabelVerticalAlignment', 'bottom');
    legend([hAvg hY hX], {'Average Response', '$63.2\%$ of $\mathrm{K}$', '$\mathrm{T}$'}, 'Interpreter', 'latex', 'Location', 'best');
else
    legend([hAvg hY], {'Average Response', '$63.2\%$ of $\mathrm{K}$'}, 'Interpreter', 'latex', 'Location', 'best');
end
hold off;
%saveas(gcf, 'C:\Users\ivadu\Desktop\8.semestrik\vymennik\prez\average_step_response.png');

%% saving data


Ytrain=y(1:2000);
Utrain=u(1:2000);

%save("train_data.mat", 'Ytrain', 'Utrain')
 

Ytest=y(2001:end);
Utest=u(2001:end);
%save("test_data.mat", 'Ytest', 'Utest')

%% 
%% Raw measurement overview
figure('Color','w','Name','Raw Identification Data');

subplot(2,1,1);
plot(t, u, 'LineWidth', 1.2);
grid on; box on;
ylabel('Pump F speed (\%)');
title('Input signal');

subplot(2,1,2);
plot(t, y, 'LineWidth', 1.2);
grid on; box on;
xlabel('Time (s)');
ylabel('Outlet temperature $T_4$ ($^\circ$C)');
title('Measured output');

%% Step detection
step_indices = find(abs(diff(u)) > 1);
num_steps = numel(step_indices);

%% Signal normalization
y_scaled = (y - y_mean) / y_std;
u_scaled = (u - u_mean) / u_std;

delay = 10;          % samples after step
max_length = 250;    % step response length

y_steps = NaN(num_steps, max_length);

%% Normalized step responses
t_end = t(end);   % common time limit

% Raw measured signals used for identification
fig1 = figure('Color','w','Name','Raw Identification Measurements');

% --- farby ---
c_light = [0.6 0.8 1.0];   % bledomodrá
c_dark  = [0 0.2 0.6];     % tmavomodrá

idx_train = 1:2000;
idx_test  = 2001:length(t);

subplot(2,1,1); hold on;
plot(t(idx_train), u(idx_train+1), 'Color', c_light, 'LineWidth', 1.2);
plot(t(idx_test),  u(idx_test),  'Color', c_dark,  'LineWidth', 1.2);
grid on; box on;
ylabel('Pump F speed (%)');
title('Input signal');
legend('Training data','Test data','Location','southwest');
ylim([0 101])
xlim([0 3000])
hold off;


subplot(2,1,2); hold on;
plot(t(idx_train), y(idx_train), 'Color', c_light, 'LineWidth', 1.2);
plot(t(idx_test),  y(idx_test),  'Color', c_dark,  'LineWidth', 1.2);
grid on; box on;
xlabel('Time (s)');
ylabel('Outlet temperature T_4 (°C)');
title('Measured output');
legend('Training data','Test data','Location','best');
hold off;
xlim([0 3000])
saveas(fig1, 'figs/fig_raw_measurementsSplit.png');


colors = lines(min(num_steps,10));

fig2 = figure('Color','w','Name','Normalized Step Responses'); 
hold on; grid on; box on;

xlabel('Time since step (s)');
ylabel('Normalized output');
title('Normalized step responses');

color_idx = 1;   % index do 'colors'

for i = 1:num_steps
    idx = step_indices(i);

    % Bounds check
    if idx-1 < 1 || idx+delay+max_length-1 > length(u)
        continue
    end

    % Zmena vstupu
    du = u_scaled(idx+delay) - u_scaled(idx-1);
    if abs(du) < eps
        continue
    end

    % Výsek výstupu okolo skoku
    y_step = y_scaled(idx:idx+max_length-1);

    % Normalizácia na jednotkový skok
    y_norm = (y_step - y_scaled(idx-1)) / abs(du);
    if du < 0
        y_norm = -y_norm;
    end

    % Uloženie do matice
    y_steps(i,:) = y_norm(:)';

    % Výber farby (cyklicky)
    c = colors(color_idx, :);
    color_idx = color_idx + 1;
    if color_idx > size(colors,1)
        color_idx = 1;
    end

    % Kreslenie
    plot(0:max_length-1, y_norm, 'Color', c, 'LineWidth', 1);
end

%saveas(fig2, 'figs/fig_normalized_step_responses.png');
%%Average normalized step response
avg_step = mean(y_steps, 1, 'omitnan');

fig3 = figure('Color','w','Name','Average Step Response');
plot(0:max_length-1, avg_step, 'k', 'LineWidth', 2);
grid on; box on;

xlabel('Time since step (s)');
ylabel('Normalized output');
title('Average normalized step response');

%%Strejc model parameter estimation
K = mean(avg_step(end-6:end), 'omitnan');
target = 0.632 * K;
tau_idx = find(avg_step >= target, 1, 'first');

hold on;
yline(target, '--', '$63.2\%$ of $K$', 'Interpreter','latex');
if ~isempty(tau_idx)
    xline(tau_idx, '--r', '$T$', 'Interpreter','latex');
end
legend('Average response','Location','best');
%saveas(fig3, 'figs/fig_average_step_response.png');



