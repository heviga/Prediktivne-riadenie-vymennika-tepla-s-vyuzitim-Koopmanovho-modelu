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
saveas(gcf, 'C:\Users\ivadu\Desktop\8.semestrik\vymennik\prez\average_step_response.png');

%% saving data


Ytrain=y(1:2000);
Utrain=u(1:2000);

save("train_data.mat", 'Ytrain', 'Utrain')
 

Ytest=y(2001:end);
Utest=u(2001:end);
save("test_data.mat", 'Ytest', 'Utest')




