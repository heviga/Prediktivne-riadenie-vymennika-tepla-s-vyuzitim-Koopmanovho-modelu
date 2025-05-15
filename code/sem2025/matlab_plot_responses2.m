%% unscaled

%% load data
clc,close all
data1 = load('ident_data_1.mat')
data2 =load("ident_data_2_9_steps.mat")

%teploty
T4_1 = data1.Temperatures{4}.Values.Data;%3501
T4_2 = data2.Temperatures{4}.Values.Data;%2700

%casy
t1 = data1.tout;%3500
t2= data2.tout;%2699

%input
u1=data1.uout(:,2);
u2=data2.uout(:,2);

%% zle merania prec
%len skoky
u1=u1(251:end); %prvych 250s je 100
t1=t1(251:end)-250;

u2=u2(251:2250); 
t2=t2(251:2250)-250;

T4_1=T4_1(251:end);
T4_2=T4_2(251:2250);
x1=T4_1;%lebo 
x2=T4_2;

figure
subplot(2,1,1)
plot(t1,u1)
grid on, box on, grid minor
subplot(2,1,2)
plot(t1,T4_1)
grid on, box on, grid minor
title('prve meranie');

figure
subplot(2,1,1)
plot(t2,u2)
grid on, box on, grid minor
subplot(2,1,2)
plot(t2,T4_2)
grid on, box on, grid minor
title('druhe meranie');
%% saving data

Ytrain=x1(t1+1);
Utrain=u1(t1+1);
% save("train_data.mat", 'Ytrain', 'Utrain')
% save("train_data.mat", 'Y', 'U')
% save("train_data_ident.mat", 'Ytrain', 'Utrain')
% 

Ytest=x2(t2+1);
Utest=u2(t2+1);
% save("test_data.mat", 'Ytest', 'Utest')
% save("test_data.mat", 'Y', 'U')
% save("test_data_ident.mat", 'Ytest', 'Utest')


% Y = x(t+1);
% U = u(t+1);
% 
% save("data.mat", 'Y', 'U')

%% hladanie stepov
step_indices_1 = find(abs(diff(u1)) >1);
step_indices_2 = find(abs(diff(u2)) >= 1);
step_indices_2 = step_indices_2 + length(t1); % shift step casy druheho merania
%merged step indices
step_indices = [step_indices_1; step_indices_2]; % combine indices

num_steps = length(step_indices_1) + length(step_indices_2);

%merge
t = [t1; t2]; 
x = [T4_1; T4_2]; %T4
u = [u1; u2];%

x_scaled = (x - x_mean) / x_std;
u_scaled = (u - u_mean) / u_std;

%store step responses
max_length = 250;
num_steps = length(step_indices);

delay = 100;
pre_window = 1;
max_length = 250;

x_steps = NaN(num_steps, max_length);
u_steps = NaN(num_steps, 1);
colors = lines(num_steps);

%% Plot normalized and true (unscaled) step responses side by side
figure; hold on;
title('Normalized Step Responses');
xlabel('Sample Index');
ylabel('Normalized Output');
grid on; grid minor;

for i = 1:num_steps
    start_idx = step_indices(i);

    % Bounds check
    if start_idx - 1 < 1 || (start_idx + delay >= length(u))
        fprintf('Skipping step %d: not enough delay margin.\n', i);
        continue
    end

    % Δu and normalization
    u_before = u_scaled(start_idx - 1);
    u_after  = u_scaled(start_idx + delay);
    delta_u = u_after - u_before;

    x_step_scaled = x_scaled(start_idx : start_idx + 249);

    x_norm = (x_step_scaled - x_scaled(start_idx - 1)) / abs(delta_u);
    if delta_u < 0
        x_norm = -x_norm;
    end

    % Store and plot
    x_steps(i, :) = x_norm(:)';
    u_steps(i) = delta_u;

    plot(0:max_length - 1, x_norm, 'Color', colors(i,:), 'LineWidth', 1.2);

    fprintf('Step %d: Δu = %.1f\n', i, delta_u);
end

% Save figure
saveas(gcf, 'C:\Users\ivadu\Desktop\8.semestrik\vymennik\prez\all_step_scaled_only.png');

valid_rows = ~any(isnan(x_steps), 2);
avg_step = mean(x_steps(valid_rows, :), 1);

valid_steps = sum(~isnan(x_steps(:,1)));
fprintf('Valid step responses: %d / %d\n', valid_steps, num_steps);

avg_step = mean(x_steps, 1, 'omitnan');

figure;
plot(0:max_length-1, avg_step, 'k', 'LineWidth', 2);
title('Average Normalized Step Response');
xlabel('Sample Index'); ylabel('Normalized x');
grid on;

% Estimate gain and time constant
K = mean(avg_step(end-6:end), 'omitnan');
target_value = 0.632 * K;
tau_idx = find(avg_step >= target_value, 1, 'first');

fprintf('\nSteady-state gain K ≈ %.4f\n', K);
if ~isempty(tau_idx)
    fprintf('Time constant (tau) ≈ %d samples\n', tau_idx);
else
    warning('Tau could not be determined: target value not reached.');
end

% Annotate plot
hold on;
yline(K, '--', 'K');
yline(target_value, '--', '63.2% of K');
if ~isempty(tau_idx)
    xline(tau_idx, '--r', 'tau');
end
legend('Average Response', 'K', '63.2% of K', 'τ','Location','best');
hold off;
saveas(gcf, 'C:\Users\ivadu\Desktop\8.semestrik\vymennik\prez\average_step_response.png');






