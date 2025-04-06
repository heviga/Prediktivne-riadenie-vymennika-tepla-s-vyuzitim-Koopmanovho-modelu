
% MATLAB mean(x): 58.3152397954
% MATLAB std(x): 9.0723114709

% MATLAB mean(u): 54.6108889572
% MATLAB std(u): 27.6293198476

%% Load Koopman model matrices
A = load('data/A_wC.npy');   % Koopman system matrix
B = load('data/B_wC.npy');   % Koopman input matrix
C = load('data/C_wC.npy');   % Decoder / output matrix
D = 0;                       % Assuming D = 0

% Convert from .npy to MATLAB format (if needed)
A = readNPY('data/A_wC.npy');
B = readNPY('data/B_wC.npy');
C = readNPY('data/C_wC.npy');

%% Load initial condition and input trajectory
load('data/scaled_data.mat');     % Contains u_scaled and x_scaled

u_scaled = u_scaled(:);           % Ensure column vector
sim_length = length(u_scaled);

% Koopman latent state dimension
nx = size(A,1);
ny = size(C,1);

% === Initial condition in latent space ===
% Assume you have a separate encoder that gives you this
% Example: use initial measured y0 (scaled), and precomputed x0
y0 = x_scaled(1);       % scaled output
x0 = C \ y0;            % Approximate inverse (⚠️ works only if C is square + invertible)

%% Run Koopman prediction
x_koopman = zeros(nx, sim_length+1);
y_koopman = zeros(ny, sim_length+1);

x_koopman(:,1) = x0;
y_koopman(:,1) = C * x0;

for t = 1:sim_length
    x_koopman(:,t+1) = A * x_koopman(:,t) + B * u_scaled(t);
    y_koopman(:,t+1) = C * x_koopman(:,t+1);
end

%% Descale Koopman output
x_mean = 46.345336;
x_std = 6.814313;
y_koopman_desc = y_koopman * x_std + x_mean;

%% Plot Koopman predicted output
time = 0:sim_length;

figure;
plot(time, y_koopman_desc, 'LineWidth', 2);
ylabel('Output y (°C)');
xlabel('Time step');
title('Koopman Model Prediction');
grid on;































