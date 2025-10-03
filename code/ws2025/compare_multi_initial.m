clc; clear; close all
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');

% Add path to readNPY function
addpath('../');

% Initialize Python environment for baseline inference
pyenv('Version', 'C:\Users\ivadu\AppData\Local\Programs\Python\Python39\python.exe');
py.sys.path().append('C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\ws2025');
py.baseline_inference.init();

%% Load models
% Koopman model (from model_matlab_koopman_cl)
A_k = double(readNPY('data/A_wC_all.npy'));
B_k = double(readNPY('data/B_wC_all.npy'));
C_k = double(readNPY('data/C_wC_all.npy'));
D_k = 0;

% Strejc model (from model_matlab_strejc_MPC)
A_s = 0.97701252;
B_s = 0.03150894;  % Updated B parameter from model_matlab_strejc_MPC
C_s = 1;
D_s = 0;

%% Load scaling data
load('train_data_ident.mat');
load('test_data_ident.mat');

Ytrain = Ytrain(:);
Utrain = Utrain(:);
Ytest = Ytest(:);
Utest = Utest(:);

Yall = [Ytrain; Ytest];
Uall = [Utrain; Utest];

x_mean = mean(Yall);
x_std = std(Yall);
u_mean = mean(Uall);
u_std = std(Uall);

%% MPC parameters
Qy = 10;
Qu = 1;
N = 40;
sim_length = 150;

umin = (20 - u_mean) / u_std;
umax = (100 - u_mean) / u_std;
ymin = (0 - x_mean) / x_std;
ymax = (70 - x_mean) / x_std;

%% Koopman MPC controller (from model_matlab_koopman_cl)
nx_k = size(A_k,1);

u_k = sdpvar(repmat(1,1,N), repmat(1,1,N));
x_k = sdpvar(repmat(nx_k,1,N+1), repmat(1,1,N+1));
x0_k = sdpvar(nx_k,1);

constraints = [x_k{1} == x0_k];
objective = 0;

for k = 1:N
    constraints = [constraints, x_k{k+1} == A_k * x_k{k} + B_k * u_k{k}];
    yk = C_k * x_k{k};
    constraints = [constraints, umin <= u_k{k} <= umax];
    constraints = [constraints, ymin <= yk <= ymax];
    objective = objective + Qy * (yk)^2 + Qu * u_k{k}^2;
end

koopman_controller = optimizer(constraints, objective, sdpsettings('solver','quadprog'), x0_k, u_k{1});

%% Strejc MPC controller (from model_matlab_strejc_MPC)
nx_s = 1;

u_s = sdpvar(repmat(1,1,N), repmat(1,1,N));
x_s = sdpvar(repmat(1,1,N+1), repmat(1,1,N+1));
x0_s = sdpvar(1,1);

constraints = [x_s{1} == x0_s];
objective = 0;

for k = 1:N
    constraints = [constraints, x_s{k+1} == A_s * x_s{k} + B_s * u_s{k}];
    y_k = C_s * x_s{k};
    constraints = [constraints, umin <= u_s{k} <= umax];
    constraints = [constraints, ymin <= y_k <= ymax];
    objective = objective + Qy * (y_k)^2 + Qu * u_s{k}^2;
end

strejc_controller = optimizer(constraints, objective, sdpsettings('solver','quadprog'), x0_s, u_s{1});

%% Kalman filter parameters (from both model files)
% Koopman KF parameters
Q_KF = 0.5*eye(nx_k);
R_KF = 0.1;
P_k = eye(nx_k);
meas_noise_std = 0;

% Strejc KF parameters  
Q_kalman = 0.5;
R_kalman = 0.1;
P_s = 1;

%% Define initial conditions
lower = linspace(50,x_mean-5,10);
higher = linspace(x_mean+5,70,10);

y0_vals = [lower,higher];
y0_vals_scaled = (y0_vals - x_mean) / x_std;

results_koopman = [];
results_strejc = [];

%% Run tests
for i = 1:length(y0_vals_scaled)
    fprintf("Simulation %d / %d\n", i, length(y0_vals_scaled));
    
    % Calculate initial conditions for this iteration
    y0_scaled = y0_vals_scaled(i);
    x0_k = pinv(C_k) * y0_scaled;  % Proper matrix multiplication
    
    % --- Koopman with Baseline Integration ---
    xk = zeros(nx_k, sim_length+1);
    yk = zeros(1, sim_length+1);
    uk = zeros(1, sim_length);
    x_est_k = zeros(nx_k, sim_length+1);
    y_meas_k = zeros(1, sim_length+1);
    y_true_k = zeros(1, sim_length+1);
    
    % Reset Kalman filter state for each iteration
    P_k = eye(nx_k);

    xk(:,1) = x0_k;
    x_est_k(:,1) = x0_k;
    yk(:,1) = C_k * x0_k;
    y_true_k(:,1) = yk(:,1);
    y_meas_k(:,1) = y_true_k(:,1) + meas_noise_std * randn(1,1);

    for t = 1:sim_length
        try
            uk(:,t) = koopman_controller{x_est_k(:,t)};
            
            % Check for NaN in control input
            if isnan(uk(:,t))
                fprintf('Warning: NaN koopman control input at t=%d, IC=%d\n', t, i);
                uk(:,t) = 0;  % Set to zero if NaN
            end
        catch ME
            fprintf('Error in Koopman controller at t=%d, IC=%d: %s\n', t, i, ME.message);
            uk(:,t) = 0;  % Set to zero if error
        end
        
        % === BASELINE INFERENCE INTEGRATION ===
        if t == 1
            % Initialize baseline model with current state for this iteration
            py.baseline_inference.get_x(y_true_k(:,t));
        end
        
        % Get measurement from baseline model (true system)
        try
            y_baseline = py.baseline_inference.y_plus(uk(:,t));
            y_baseline_array = double(y_baseline);
            y_true_k(:,t+1) = y_baseline_array(1);
            
            % Check for NaN in baseline output
            if isnan(y_true_k(:,t+1))
                fprintf('Warning: NaN baseline output at t=%d, IC=%d\n', t, i);
                y_true_k(:,t+1) = y_true_k(:,t);  % Keep previous value
            end
        catch ME
            fprintf('Error in baseline inference at t=%d, IC=%d: %s\n', t, i, ME.message);
            y_true_k(:,t+1) = y_true_k(:,t);  % Keep previous value
        end
        
        y_meas_k(:,t+1) = y_true_k(:,t+1) + meas_noise_std * randn(1,1);
        
        % Kalman filter update
        x_pred = A_k * x_est_k(:,t) + B_k * uk(:,t);
        P_pred = A_k * P_k * A_k' + Q_KF;
        
        S = C_k * P_pred * C_k' + R_KF;
        K = (P_pred * C_k') / S;
        x_est_k(:,t+1) = x_pred + K * (y_meas_k(:,t+1) - C_k * x_pred);
        P_k = (eye(nx_k) - K * C_k) * P_pred;
        P_k = (P_k + P_k')/2;
        
        % Model prediction for comparison
        xk(:,t+1) = A_k * xk(:,t) + B_k * uk(:,t);
        yk(:,t+1) = C_k * xk(:,t+1);
    end

    yk_desc = y_true_k * x_std + x_mean;
    uk_desc = uk * u_std + u_mean;
    e_k = yk_desc(1:end-1);

    u_sum_koop = sum(abs(uk_desc));
    y_sum_koop = sum(abs(e_k));
    e_sum_koop_y = sum(abs(e_k - x_mean));
    e_sum_koop_u = sum(abs(uk_desc - u_mean));
    rmse_koop = sqrt(mean((e_k-x_mean).^2));
    obj_koop = sum(Qy * (y_true_k(1:end-1)).^2 + Qu * (uk).^2);

    koopman_metrics = [u_sum_koop, y_sum_koop, e_sum_koop_y, e_sum_koop_u, rmse_koop, obj_koop];
    results_koopman = [results_koopman; koopman_metrics];

    % --- Strejc with Baseline Integration ---
    xs = zeros(1, sim_length+1);
    ys = zeros(1, sim_length+1);
    us = zeros(1, sim_length);
    x_est_s = zeros(1, sim_length+1);
    y_meas_s = zeros(1, sim_length+1);
    y_true_s = zeros(1, sim_length+1);
    
    % Reset Kalman filter state for each iteration
    P_s = 1;

    xs(:,1) = y0_vals_scaled(i);
    x_est_s(:,1) = y0_vals_scaled(i);
    ys(:,1) = C_s * xs(:,1);
    y_true_s(:,1) = ys(:,1);
    y_meas_s(:,1) = y_true_s(:,1) + meas_noise_std * randn(1,1);

    for t = 1:sim_length
        try
            us(:,t) = strejc_controller{x_est_s(:,t)};
            
            % Check for NaN in control input
            if isnan(us(:,t))
                fprintf('Warning: NaN strejc control input at t=%d, IC=%d\n', t, i);
                us(:,t) = 0;  % Set to zero if NaN
            end
        catch ME
            fprintf('Error in Strejc controller at t=%d, IC=%d: %s\n', t, i, ME.message);
            us(:,t) = 0;  % Set to zero if error
        end
        
        % === BASELINE INFERENCE INTEGRATION ===
        if t == 1
            % Initialize baseline model with current state for this iteration
            py.baseline_inference.get_x(y_true_s(:,t));
        end
        
        % Get measurement from baseline model (true system)
        try
            y_baseline = py.baseline_inference.y_plus(us(:,t));
            y_baseline_array = double(y_baseline);
            y_true_s(:,t+1) = y_baseline_array(1);
            
            % Check for NaN in baseline output
            if isnan(y_true_s(:,t+1))
                fprintf('Warning: NaN baseline output at t=%d, IC=%d\n', t, i);
                y_true_s(:,t+1) = y_true_s(:,t);  % Keep previous value
            end
        catch ME
            fprintf('Error in baseline inference at t=%d, IC=%d: %s\n', t, i, ME.message);
            y_true_s(:,t+1) = y_true_s(:,t);  % Keep previous value
        end
        
        y_meas_s(:,t+1) = y_true_s(:,t+1) + meas_noise_std * randn(1,1);
        
        % Kalman filter update
        x_pred = A_s * x_est_s(:,t) + B_s * us(:,t);
        P_pred = A_s * P_s * A_s' + Q_kalman;
        
        S = C_s * P_pred * C_s' + R_kalman;
        if abs(S) < 1e-10  % Avoid division by very small numbers
            K = 0;
        else
            K = (P_pred * C_s') / S;
        end
        x_est_s(:,t+1) = x_pred + K * (y_meas_s(:,t+1) - C_s * x_pred);
        P_s = (1 - K * C_s) * P_pred;
        P_s = max(P_s, 1e-6);  % Ensure positive definiteness
        
        % Model prediction for comparison
        xs(:,t+1) = A_s * xs(:,t) + B_s * us(:,t);
        ys(:,t+1) = C_s * xs(:,t+1);
    end

    ys_desc = y_true_s * x_std + x_mean;
    us_desc = us * u_std + u_mean;
    e_s = ys_desc(1:end-1);

    % Check for NaN values in results
    if any(isnan(us_desc)) || any(isnan(ys_desc))
        fprintf('Warning: NaN values detected in Strejc results for IC %d\n', i);
        % Set to zero or previous valid values
        us_desc(isnan(us_desc)) = 0;
        ys_desc(isnan(ys_desc)) = x_mean;
        e_s = ys_desc(1:end-1);
    end

    u_sum_strejc = sum(abs(us_desc));
    y_sum_strejc = sum(abs(e_s));
    e_sum_strejc_y = sum(abs(e_s-x_mean));
    e_sum_strejc_u = sum(abs(us_desc-u_mean));
    rmse_strejc = sqrt(mean((e_s-x_mean).^2));
    obj_strejc = sum(Qy * (y_true_s(1:end-1)).^2 + Qu * (us).^2);

    % Final check for NaN in metrics
    if any(isnan([u_sum_strejc, y_sum_strejc, e_sum_strejc_y, e_sum_strejc_u, rmse_strejc, obj_strejc]))
        fprintf('Warning: NaN metrics for Strejc IC %d, setting to zero\n', i);
        u_sum_strejc = 0; y_sum_strejc = 0; e_sum_strejc_y = 0; e_sum_strejc_u = 0; rmse_strejc = 0; obj_strejc = 0;
    end

    strejc_metrics = [u_sum_strejc, y_sum_strejc, e_sum_strejc_y, e_sum_strejc_u, rmse_strejc, obj_strejc];
    results_strejc = [results_strejc; strejc_metrics];
end

%% Results (Pretty Print)
fprintf('\n--- Koopman Results (sum|u| sum|y| sum|e|(y) sum|e|(u) RMSE Obj) ---\n');
for i = 1:size(results_koopman,1)
    fprintf('IC #%2d (y0=%.2f°C):  sum|u|=%.2f  sum|y|=%.2f  sum|e|(y)=%.2f  sum|e|(u)=%.2f  RMSE=%.2f  Obj=%.2f\n', ...
        i, y0_vals(i), results_koopman(i,1), results_koopman(i,2), results_koopman(i,3), results_koopman(i,4), results_koopman(i,5), results_koopman(i,6));
end

fprintf('\n--- Strejc Results (sum|u| sum|y| sum|e|(y) sum|e|(u) RMSE Obj) ---\n');
for i = 1:size(results_strejc,1)
    fprintf('IC #%2d (y0=%.2f°C):  sum|u|=%.2f  sum|y|=%.2f  sum|e|(y)=%.2f  sum|e|(u)=%.2f  RMSE=%.2f  Obj=%.2f\n', ...
        i, y0_vals(i), results_strejc(i,1), results_strejc(i,2), results_strejc(i,3), results_strejc(i,4), results_strejc(i,5), results_strejc(i,6));
end

save('results_20x0_MPC_ws2025.mat', 'results_koopman', 'results_strejc', 'y0_vals');

%% Visualization of Metrics
metric_names = {'RMSE', 'Objective function'};
metric_idx = [5, 6];

% First 10 initial conditions
figure('Name','First 10 initial conditions','Position',[100 100 800 400]);
for i = 1:2
    subplot(1,2,i)
    plot(y0_vals(1:10), results_koopman(1:10,metric_idx(i)), 'm-o','LineWidth',2); hold on;
    plot(y0_vals(1:10), results_strejc(1:10,metric_idx(i)), 'b-s','LineWidth',2);

    xlabel('$\mathrm{Initial\ condition\ } y_0\ (^\circ\mathrm{C})$', 'Interpreter', 'latex');
    if i == 1
        ylabel('RMSE ($^\circ$C)', 'Interpreter', 'latex');
    else
        ylabel('Objective function', 'Interpreter', 'latex');
    end
    title(metric_names{i}, 'Interpreter', 'latex');
    legend('Koopman','Strejc','Location','Best', 'Interpreter','latex');
    set(gca, 'FontName', 'Latin Modern Roman');
    grid on; grid minor;
end
sgtitle('Control with Initial Conditions Below Steady-State');

saveas(gcf, 'metrics_rmse_obj_first10_ws2025.png');

% Last 10 initial conditions
figure('Name','Last 10 initial conditions','Position',[100 100 800 400]);
for i = 1:2
    subplot(1,2,i)
    plot(y0_vals(11:20), results_koopman(11:20,metric_idx(i)), 'm-o','LineWidth',2); hold on;
    plot(y0_vals(11:20), results_strejc(11:20,metric_idx(i)), 'b-s','LineWidth',2);

    xlabel('$\mathrm{Initial\ condition\ } y_0\ (^\circ\mathrm{C})$', 'Interpreter', 'latex');
    if i == 1
        ylabel('RMSE ($^\circ$C)', 'Interpreter', 'latex');
    else
        ylabel('Objective function', 'Interpreter', 'latex');
    end
    title(metric_names{i}, 'Interpreter', 'latex');
    legend('Koopman','Strejc','Location','Best', 'Interpreter','latex');
    set(gca, 'FontName', 'Latin Modern Roman');
    grid on; grid minor;
end
sgtitle('Control with Initial Conditions Above Steady-State');

saveas(gcf, 'metrics_rmse_obj_last10_ws2025.png');

%% Summary of metrics
metrics_labels = {'Sum |u|', 'Sum |y|', 'Sum e (y)', 'Sum e (u)', 'RMSE', 'Objective'};

koopman_mean = mean(results_koopman);
strejc_mean = mean(results_strejc);

koopman_median = median(results_koopman);
strejc_median = median(results_strejc);

koopman_std = std(results_koopman);
strejc_std = std(results_strejc);

fprintf('\n========== Koopman vs Strejc - Statistics Summary ==========\n');

for i = 1:length(metrics_labels)
    fprintf('\nMetric: %s\n', metrics_labels{i});
    
    fprintf('  Koopman -> Mean: %.2f, Median: %.2f, Std: %.2f\n', ...
        koopman_mean(i), koopman_median(i), koopman_std(i));
    
    fprintf('  Strejc  -> Mean: %.2f, Median: %.2f, Std: %.2f\n', ...
        strejc_mean(i), strejc_median(i), strejc_std(i));
end
