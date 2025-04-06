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
num_steps = length(step_indices_1) + length(step_indices_2);

%merge
t = [t1; t2]; 
x = [T4_1; T4_2]; %T4
u = [u1; u2];%

%merged step indices
step_indices_2 = step_indices_2 + length(t1); % shift step casy druheho merania
step_indices = [step_indices_1; step_indices_2]; % combine indices

%store step responses
max_length = 250;
x_steps = NaN(num_steps, max_length);
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
        end_idx = step_indices(end)+250;
    end

    % Extract step response and reset time
    x_step = x(start_idx:end_idx);
    t_step = t(start_idx:end_idx) - t(start_idx);%0-250
    
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
    disp(['Step ', num2str(i), ': Δu = ', num2str(step_size)]);

    plot(0:len-1, x_steps(i, 1:len), 'Color', colors(i,:));
end

% Plot formatting
title('Step Responses');
xlabel('Index');
ylabel('Normalized x');
grid on;
hold off;

% priemerna step response
avg_step = nanmean(x_steps, 1);

figure;
plot(0:max_length-1, avg_step, 'k', 'LineWidth', 2);
title('Average Normalized Step Response');
xlabel('Index');
ylabel('Normalized x');
grid on;

% === Compute gain K and time constant tau ===
K = mean(avg_step(end-5:end));           % Steady-state gain
target_value = 0.632 * K;                % 63.2% of K for first-order system

% Find index where the average response reaches or exceeds 63.2% of K
tau_idx = find(avg_step >= target_value, 1, 'first');

fprintf('Steady-state gain K ≈ %.4f\n', K);
fprintf('Time constant (tau) ≈ %d samples\n', tau_idx);

% Optional: plot tau visually
hold on;
yline(K, '--', 'K');
yline(target_value, '--', '63.2% of K');
xline(tau_idx, '--r', 'tau');
legend('Average Response', 'K', '63.2% of K', 'tau');
hold off;




% %vektory x,u
% x_min = min(x);
% x_max = max(x);
% u_min = min(u);
% u_max = max(u);
% 
% % normalizovane
% x_norm = minmax_normalize(x, x_min, x_max);
% u_norm = minmax_normalize(u, u_min, u_max);



%% scaling vektorov
function x_norm = minmax_normalize(x, xmin, xmax)
    x_norm = (x - xmin) / (xmax - xmin);
end







