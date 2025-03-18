clc; close all; clear;

% Load data
data1 = load('ident_data_1.mat');
data2 = load("ident_data_2_9_steps.mat");

% Extract temperature (T4) and time
T4_1 = data1.Temperatures{4}.Values.Data;
T4_2 = data2.Temperatures{4}.Values.Data;
t1 = data1.tout;
t2 = data2.tout;

% Extract input signal
u1 = data1.uout(:,2);
u2 = data2.uout(:,2);

% Merge data
t = [t1; t2]; 
x = [T4_1; T4_2]; 
u = [u1; u2];

% Find step indices every 250s
step_indices = find(mod(t, 250) == 0);

% Remove duplicate/invalid indices
step_indices = unique(step_indices);
num_steps = length(step_indices);

% Define step duration (250 samples per step)
step_duration = 250; 
max_length = step_duration;
x_steps = NaN(num_steps, max_length);

% Plot step responses
figure; hold on;
colors = lines(num_steps);

for i = 1:num_steps
    start_idx = step_indices(i);
    
    % Define end of step (250 samples later, or end of data)
    end_idx = min(start_idx + step_duration - 1, length(t));
    
    % Extract step response
    x_step = x(start_idx:end_idx);
    t_step = t(start_idx:end_idx) - t(start_idx);
    
    % Step size
    step_size = u(start_idx+1) - u(start_idx);
    
    if step_size ~= 0  % Avoid division by zero
        x_norm = (x_step - x_step(1)) / abs(step_size);
        
        % Ensure positive step response
        if step_size < 0
            x_norm = -x_norm;
        end
        
        % Store normalized step response
        len = length(x_norm);
        x_steps(i, 1:len) = x_norm;
        
        % Plot
        plot(0:len-1, x_steps(i, 1:len), 'Color', colors(i,:));
    end
end

title('Normalized Step Responses (Every 250s)');
xlabel('Index');
ylabel('Normalized x');
grid on;
hold off;

%% Normalization of x and u vectors
x_norm = minmax_normalize(x, min(x), max(x));
u_norm = minmax_normalize(u, min(u), max(u));

%% Normalization Function
function x_norm = minmax_normalize(x, xmin, xmax)
    if xmin ~= xmax  % Avoid division by zero
        x_norm = (x - xmin) / (xmax - xmin);
    else
        x_norm = zeros(size(x)); % If all values are the same, return zero vector
    end
end
