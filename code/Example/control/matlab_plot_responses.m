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


%kedy step change
step_indices_1 = find(abs(diff(u1)) > 10);
step_indices_2 = find(abs(diff(u2)) > 10);
num_steps = length(step_indices_1) + length(step_indices_2);

%merge
t = [t1; t2]; 
x = [T4_1; T4_2]; %T4
u = [u1; u2];

%merged step indices
step_indices_2 = step_indices_2 + length(t1); % shift step casy druheho merania
step_indices = [step_indices_1; step_indices_2]; % combine indices

%store step responses
max_length = 250;
x_steps = NaN(num_steps, max_length);


%vyhodime posledny skoro nulovy step change
step_durations = [diff(step_indices); length(t) - step_indices(end)];
short_step_idx = find(step_durations == 200, 1, 'last'); % Find the last occurrence
if ~isempty(short_step_idx)
    step_indices(short_step_idx) = [];
end
num_steps = length(step_indices);

figure; hold on;
colors = lines(num_steps); % Generate different colors


for i = 1:num_steps
    % Get step start index
    start_idx = step_indices(i);
    
    % Define end of step (next step or end of data)
    if i < num_steps
        end_idx = step_indices(i+1) - 1;
    else
        end_idx = length(t);
    end

    % Extract step response and reset time
    x_step = x(start_idx:end_idx);
    t_step = t(start_idx:end_idx) - t(start_idx);
    
    % Compute step change (difference in u)
    step_size = u(start_idx+1) - u(start_idx);
    
    % Normalize relative to step change
    x_norm = (x_step - x_step(1)) / abs(step_size);  

    % If step change is negative, flip response to be positive
    if step_size < 0
        x_norm = -x_norm;
    end

    % Store in matrix (truncate or pad with NaN)
    len = min(length(x_norm), max_length);
    x_steps(i, 1:len) = x_norm(1:len);

    % Plot each step response
    plot(0:len-1, x_steps(i, 1:len), 'Color', colors(i,:));
end

% Plot formatting
title('Normalized Step Responses');
xlabel('Index');
ylabel('Normalized x');
grid on;
hold off;



%vektory x,u
x_min = min(x);
x_max = max(x);
u_min = min(u);
u_max = max(u);

% normalizovane
x_norm = minmax_normalize(x, x_min, x_max);
u_norm = minmax_normalize(u, u_min, u_max);







%% scaling vektorov
function x_norm = minmax_normalize(x, xmin, xmax)
    x_norm = (x - xmin) / (xmax - xmin);
end





