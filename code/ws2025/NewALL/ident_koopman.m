%mean 58.3377  torch.save(best_model, "./data/model_baseline.pth")
clear all, %close all

% Add path to readNPY function
addpath('../');
% pre readnpy 
%addpath 'C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code'

%% Load Koopman model matrices
A = readNPY('data/A_wC.npy');%neulozila som ako.m
B = readNPY('data/B_wC.npy');
C = readNPY('data/C_wC.npy');
D = 0;

%% Load unscaled training + test data
load('train_data.mat');   % Ytrain, Utrain
load('test_data.mat');    % Ytest, Utest

% Reshape if needed
Ytrain = Ytrain(:);
Utrain = Utrain(:);
Ytest = Ytest(:);
Utest = Utest(:);
Yall = [Ytrain; Ytest];
Uall = [Utrain; Utest];


%% Scale using only training data stats
x_mean = mean(Yall);
x_std = std(Yall);

u_mean = mean(Uall);
u_std = std(Uall);

Ytest_scaled = (Ytest - x_mean) / x_std;
Utest_scaled = (Utest - u_mean) / u_std;

%% Initial condition from test data (scaled)
y0 = Ytest_scaled(1);         % first scaled output from test set
%x0 = C \ y0;   
x0 = pinv(C) * y0;             % least-squares solution C*x0 ≈ y0
% x0 = zeros(size(A,1),1); 
% x0(1) = y0;    % Approximate latent state (assumes C is invertible)

%% Koopman rollout
sim_length = length(Utest_scaled);
nx = size(A,1);
ny = 1;

x_koopman = zeros(nx, sim_length+1);
y_koopman = zeros(ny, sim_length+1);

x_koopman(:,1) = x0;
y_koopman(:,1) = C * x0;

for t = 1:sim_length
    x_koopman(:,t+1) = A * x_koopman(:,t) + B * Utest_scaled(t);
    y_koopman(:,t+1) = C * x_koopman(:,t+1);
end

%% Descale output
y_koopman_desc = y_koopman * x_std + x_mean;

%% Open-loop RMSE (Koopman model)

% Align lengths (skip initial condition)
% y_model_ol = y_koopman_desc(2:end);   % Koopman prediction
% y_meas     = Ytest(:);                % True measured output

% % Make sure shapes match
% y_model_ol = y_model_ol(:);
% y_meas     = y_meas(:);

% RMSE computation
%% Open-loop RMSE (Koopman model) — CORRECT

% Remove initial condition and align lengths
y_model_ol = y_koopman_desc(2:end);   % 1001 samples
y_meas     = Ytest;                   % 1001 samples

% Force column vectors
y_model_ol = y_model_ol(:);
y_meas     = y_meas(:);

% Final sanity check
assert(all(size(y_model_ol) == size(y_meas)), 'Signals not aligned!');

% RMSE
rmse_koopman = sqrt(mean((y_model_ol - y_meas).^2));

fprintf('Open-loop RMSE (Koopman model): %.3f °C\n', rmse_koopman);

%% Plot
time = 0:sim_length;

hold on
subplot(2,1,1)
plot(time(1:end-1), y_koopman_desc(1:end-1), 'LineWidth', 2); hold on;
plot(time(1:end-1), Ytest, '--k', 'LineWidth', 1.5);
xlabel('Time step');
ylabel('Output y (°C)');
title('Koopman Prediction vs True Output');
legend('Koopman prediction', 'True test data');
grid on;

subplot(2,1,2)
stairs(time(1:end-1), Utest, 'LineWidth', 2);
xlabel('Time step');
ylabel('Input u');
title('Original Input (Test Data)');
grid on;
%%
% save('koopman_ol.mat', ...
 %    'y_koopman_desc', 'Ytest', 'time', 'Utest');
