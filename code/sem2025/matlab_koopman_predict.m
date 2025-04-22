%PRE TRAIN SET
% MATLAB mean(x): 58.3152397954
% MATLAB std(x): 9.0723114709

% MATLAB mean(u): 54.6108889572
% MATLAB std(u): 27.6293198476
clear all, %close all
%% Load Koopman model matrices
A = readNPY('data/A_wC_all.npy');%neulozila som ako.m
B = readNPY('data/B_wC_all.npy');
C = readNPY('data/C_wC_all.npy');
D = 0;

%% Load unscaled training + test data
load('data/train_data_ident.mat');   % Ytrain, Utrain
load('data/test_data_ident.mat');    % Ytest, Utest

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
x0 = zeros(size(A,1),1); 
x0(1) = y0;    % Approximate latent state (assumes C is invertible)

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

save('koopman_open_loop_comparison.mat', ...
    'y_koopman_desc', 'Ytest', 'time', 'Utest');
