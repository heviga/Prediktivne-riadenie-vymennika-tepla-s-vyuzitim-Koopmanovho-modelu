clc,close all, clear all
data = load('identifikacia2.mat'); %3000
%prvy zly skok prec 
%finale 3000 samples
y = data.Temperatures{4}.Values.Data;% temperatures t4
t = data.tout;
u=data.uout(:,2);
u=u(251:end);
y=y(251:end);
t=t(251:end);
% reset time to start from 1
t = t - t(1) + 1;

%% save full data we use 
%save('../data/full_data.mat', 'u', 'y', 't');

y_mean = mean(y); % 58.3377
u_mean = mean(u); % 63.0883
y_std = std(y); % 7.1204
u_std = std(u); % 23.9910

t_end = t(end);   % common time limit

% Raw measured signals used for identification
fig1 = figure('Color','w','Name','Raw Identification Measurements');

% --- farby ---
c_light = [0.6 0.8 1.0];   % bledomodrá
c_dark  = [0 0.2 0.6];     % tmavomodrá

% nove rozdelenie 2 skoky dole sva hore (8.-11.)

idx_train = 1:1999;
idx_test  = 2000:length(t);

subplot(2,1,1); hold on;
plot(t(idx_train), u(idx_train), 'Color', c_light, 'LineWidth', 1.2);
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

% average scaled step v stepsCropped.m
%% test,train
ytrain = y(idx_train);
ttrain = t(idx_train);
utrain = u(idx_train);

ytest = y(idx_test);
utest = u(idx_test);
ttest = t(idx_test);

y_scaled_test = (ytest - y_mean) / y_std;
u_scaled_test = (utest - u_mean) / u_std;
%%

% % ===== OUTPUT (T4) =====
% subplot(2,1,1)
% plot(ttest, ytest, 'LineWidth', 1.3);
% grid on;
% ylabel('T4 [°C]');
% title('Test data – Output');
% xlim([ttest(1) ttest(end)])
% 
% % ===== INPUT (Pump2) =====
% subplot(2,1,2)
% plot(ttest, utest, 'LineWidth', 1.3);
% grid on;
% ylabel('Pump2 [%]');
% xlabel('Time [s]');
% title('Test data – Input');
% xlim([ttest(1) ttest(end)])
%% Strejc 
K = 1.1237;
tau = 68;
Ts = 1;

% Discrete-time Strejc model
A = exp(-Ts/tau); %0.985401721021654
B = K*(1 - A);%0.016404086087968
C = 1;
D = 0;

Ts = 1;             % Sampling time
nx = 1;             % Number of states
nu = 1;             % Number of inputs
ny = 1;             % Number of outputs

% Simulation open loop
sim_steps = length(utest);

x_open = zeros(nx, sim_steps+1);
y_open = zeros(ny, sim_steps+1);

x_open(:,1) = y_scaled_test(1);
y_open(:,1) = C * x_open(:,1);

for t = 1:sim_steps
    x_open(:,t+1) = A * x_open(:,t) + B * u_scaled_test(t);
    y_open(:,t+1) = C * x_open(:,t+1);
end

time = 0:sim_steps;
%descale to plot
u_open_desc = u_scaled_test * u_std + u_mean;
y_open_desc = y_open * y_std + y_mean;

y_strejc_pred = y_open_desc(1:end-1);

figure;
subplot(2,1,1)
plot(time, y_open_desc, 'b-', 'LineWidth', 2); hold on
plot(time(1:end-1), ytest, 'k:', 'LineWidth', 1.5);
xlabel('Time step'); ylabel('Output y (°C)');
legend('Open-loop (Strejc)', 'True Output');
title('Output comparison');
grid on;
xlim([1 sim_steps])

subplot(2,1,2)
stairs(time(1:end-1), u_open_desc, 'r-', 'LineWidth', 2); hold on;
xlabel('Time step'); ylabel('Input u');
legend('Open-loop Input');
title('Input Comparison');
grid on; grid minor;
xlim([1 sim_steps])
%% koopman
A_koop = readNPY('../data/A_wC.npy');%neulozila som ako.m
B_koop = readNPY('../data/B_wC.npy');
C_koop = readNPY('../data/C_wC.npy');
D_koop = 0;

%koopman openloop

% initial cond
y0_koop = y_scaled_test(1);
x0_koop = pinv(C_koop) * y0_koop; %pseudoinverse

nx_koop = size(A_koop,1); 
ny_koop = 1;

%nulove matice a prvy stlpec
x_koopman = zeros(nx_koop, sim_steps+1);
y_koopman = zeros(ny_koop, sim_steps+1);

x_koopman(:,1) = x0_koop;
y_koopman(:,1) = C_koop * x0_koop;

for t = 1:sim_steps
    x_koopman(:,t+1) = A_koop * x_koopman(:,t) + B_koop * u_scaled_test(t);
    y_koopman(:,t+1) = C_koop * x_koopman(:,t+1);
end

% descale otput
y_koopman_desc = y_koopman * y_std + y_mean;
y_koop_pred   = y_koopman_desc(1:end-1);

% ploot

figure
subplot(2,1,1)
plot(time(1:end-1), y_koopman_desc(1:end-1), 'LineWidth', 2); hold on;
plot(time(1:end-1), ytest, '--k', 'LineWidth', 1.5);
xlabel('Time step');
ylabel('Output y (°C)');
title('Koopman Prediction vs True Output');
legend('Koopman prediction', 'True test data');
grid on;
xlim([1 sim_steps])

subplot(2,1,2)
stairs(time(1:end-1), utest, 'LineWidth', 2);
xlabel('Time step');
ylabel('Input u');
title('Original Input (Test Data)');
grid on;
xlim([1 sim_steps])
%% rmse compare
y_true        = ytest(:);

rmse_strejc = sqrt(mean((y_true(:) - y_strejc_pred(:)).^2));
rmse_koop   = sqrt(mean((y_true(:) - y_koop_pred(:)).^2));

fprintf('\n===== Open-loop RMSE =====\n');
fprintf('Strejc RMSE  = %.4f °C\n', rmse_strejc);
fprintf('Koopman RMSE = %.4f °C\n', rmse_koop);

%% pllot compare
figure('Color','w');

plot(time(1:end-1), y_true, 'k', 'LineWidth', 1.8); hold on;
plot(time(1:end-1), y_strejc_pred, 'b', 'LineWidth', 1.6);
plot(time(1:end-1), y_koop_pred, 'r', 'LineWidth', 1.6);

grid on;
xlabel('Time step');
ylabel('Outlet temperature T_4 (°C)');
title('Open-loop Model Comparison on Test Data');
xlim([0 1001])

legend('True','Strejc','Koopman','Location','best');