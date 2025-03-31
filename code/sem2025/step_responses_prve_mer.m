clc, close all;
data1 = load('ident_data_1.mat');
data2 = load("ident_data_2_9_steps.mat");

% teploty
T4_1 = data1.Temperatures{4}.Values.Data; % 3501
T4_2 = data2.Temperatures{4}.Values.Data; % 2700

% casy
t1 = data1.tout; % 3500
t2 = data2.tout; % 2699

% input
u1 = data1.uout(:,2);
u2 = data2.uout(:,2);
% u2=u2(1:5750)

% kedy step change
step_indices_1 = find(abs(diff(u1)) > 1);
step_indices_1 = step_indices_1(2:end);
step_indices_2 = find(abs(diff(u2)) >= 1);
step_indices_2 = step_indices_2(2:end-2);
num_steps = length(step_indices_1) + length(step_indices_2);

% merge
t = [t1; t2];
x = [T4_1; T4_2]; % T4
u = [u1; u2];

% merged step indices
step_indices_2 = step_indices_2 + length(t1); % shift step casy druheho merania
step_indices = [step_indices_1; step_indices_2]; % combine indices

num_steps = length(step_indices);

for i = 1:num_steps
    % Get step start index
    start_idx = step_indices(i);
    
    % Define end of step (next step or end of data)
    if i < num_steps
        end_idx = step_indices(i+1) - 1;
    else
        end_idx = min(start_idx + 250, length(t));
    end

    % Extract step response and reset time
    x_step = x(start_idx:end_idx);
    t_step = t(start_idx:end_idx) - t(start_idx);
    u_step = u(start_idx:end_idx);
    
    % Plot each step response separately
    figure;
    subplot(2,1,1);
    plot(t_step, x_step, 'b', 'LineWidth', 1.5);
    title(['Step Response ' num2str(i)]);
    xlabel('Time (s)');
    ylabel('x (Temperature)');
    grid on;
    
    subplot(2,1,2);
    plot(t_step, u_step, 'r', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('u (Input)');
    grid on;
end