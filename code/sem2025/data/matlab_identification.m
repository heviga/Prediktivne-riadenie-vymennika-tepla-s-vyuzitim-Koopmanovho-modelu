clc; clear; close all;

% === Load and preprocess ===
data1 = load('ident_data_1.mat');
data2 = load('ident_data_2_9_steps.mat');

T4_1 = data1.Temperatures{4}.Values.Data;
T4_2 = data2.Temperatures{4}.Values.Data;

t1 = data1.tout;
t2 = data2.tout;

u1 = data1.uout(:, 2);
u2 = data2.uout(:, 2);

% === Crop step sequences ===
u1 = u1(251:end); t1 = t1(251:end) - 250;
u2 = u2(251:2250); t2 = t2(251:2250) - 250;

T4_1 = T4_1(251:end);
T4_2 = T4_2(251:2250);

x = [T4_1; T4_2];  % full output
u = [u1; u2];      % full input
t = [t1; t2];      % full time

% === Apply StandardScaler logic ===
x_mean = mean(x);    x_std = std(x);
u_mean = mean(u);    u_std = std(u);

x_scaled = (x - x_mean) / x_std;
u_scaled = (u - u_mean) / u_std;

% === Detect step indices ===
step_indices_1 = find(abs(diff(u1)) > 1);
step_indices_2 = find(abs(diff(u2)) > 1) + length(u1);
step_indices = [step_indices_1; step_indices_2];

% === Normalize steps ===
max_length = 250;
num_steps = length(step_indices);
x_steps = NaN(num_steps, max_length);

for i = 1:num_steps
    idx = step_indices(i);
    if i < num_steps
        next_idx = step_indices(i+1);
    else
        next_idx = idx + max_length;
    end

    x_step = x_scaled(idx:next_idx);
    if length(x_step) < 2
        continue
    end

    step_size = u(idx+1) - u(idx);
    if step_size == 0
        continue
    end

    delta_u_scaled = step_size / u_std;

    x_norm = (x_step - x_step(1)) / abs(delta_u_scaled);
    if step_size < 0
        x_norm = -x_norm;
    end

    len_valid = min(length(x_norm), max_length);
    x_steps(i, 1:len_valid) = x_norm(1:len_valid);
end

% === Average step response and time constant ===
avg_step = nanmean(x_steps, 1);
K = mean(avg_step(end-10:end));
target_value = 0.632 * K;
tau_idx = find(avg_step >= target_value, 1, 'first');

% === Plot results ===
figure;
plot(avg_step, 'k', 'LineWidth', 2); hold on;
yline(K, '--', 'K');
yline(target_value, '--', '63.2% of K');
xline(tau_idx, '--r', 'tau');
xlabel('Sample Index'); ylabel('Normalized Output');
title('Avg Normalized Step Response (Scaled x & Δu)');
grid on;
legend('Avg Step', 'K', '63.2% K', 'τ');

% === Output ===
fprintf('Steady-state gain K ≈ %.4f\n', K);
fprintf('Time constant τ ≈ %d samples\n', tau_idx);


% Compute StandardScaler-style stats
x_mean = mean(x);
x_std = std(x);
x_scaled = (x - x_mean) / x_std;

u_mean = mean(u);
u_std = std(u);
u_scaled = (u - u_mean) / u_std;

% Confirm post-scaling stats
fprintf('MATLAB x_scaled mean = %.4f, std = %.4f\n', mean(x_scaled), std(x_scaled));
fprintf('MATLAB u_scaled mean = %.4f, std = %.4f\n', mean(u_scaled), std(u_scaled));


