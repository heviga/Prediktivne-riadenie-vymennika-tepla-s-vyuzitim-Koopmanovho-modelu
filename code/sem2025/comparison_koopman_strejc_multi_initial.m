clc; clear; close all

%% Load models
A_k = double(readNPY('data/A_wC_all.npy'));
B_k = double(readNPY('data/B_wC_all.npy'));
C_k = double(readNPY('data/C_wC_all.npy'));
D_k = 0;

A_s = 0.97701252;
B_s = 0.03018256;
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

%% Koopman MPC controller (same as matlab_koopman_cl)
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

%% Strejc MPC controller (same as matlab_strejc_MPC)
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

%% Define initial conditions
lower = linspace(50,x_mean-5,10);
higher = linspace(x_mean+5,70,10);

y0_vals = [lower,higher];
y0_vals_scaled = (y0_vals - x_mean) / x_std;
% x0 = pinv(C)*y0;

results_koopman = [];
results_strejc = [];

%% Run tests
x0 = pinv(C_k).*y0_vals_scaled;
for i = 1:length(y0_vals_scaled)
    fprintf("Simulation %d / %d\n", i, length(y0_vals_scaled));
    % --- Koopman ---
    xk = zeros(nx_k, sim_length+1);
    yk = zeros(1, sim_length+1);
    uk = zeros(1, sim_length);

    xk(:,1) = x0(:,i);
    yk(:,1) = C_k* x0(:,i);


    for t = 1:sim_length
        uk(:,t) = koopman_controller{xk(:,t)};
        xk(:,t+1) = A_k * xk(:,t) + B_k * uk(:,t);
        yk(:,t+1) = C_k * xk(:,t+1);
    end

    yk_desc = yk * x_std + x_mean;
    uk_desc = uk * u_std + u_mean;
    e_k = yk_desc(1:end-1);

 

    u_sum_koop = sum(abs(uk_desc));
    y_sum_koop = sum(abs(e_k));
    e_sum_koop_y = sum(abs(e_k - x_mean));
    e_sum_koop_u = sum(abs(uk_desc - u_mean));
    rmse_koop = sqrt(mean(e_k.^2));
    obj_koop = sum(Qy * (e_k).^2 + Qu * (uk_desc).^2);

    koopman_metrics = [u_sum_koop, y_sum_koop, e_sum_koop_y, e_sum_koop_u, rmse_koop, obj_koop];
    results_koopman = [results_koopman; koopman_metrics];

    % --- Strejc ---
    xs = zeros(1, sim_length+1);
    ys = zeros(1, sim_length+1);
    us = zeros(1, sim_length);

    xs(:,1) = y0_vals_scaled(i);
    ys(:,1) = C_s * xs(:,1);

    for t = 1:sim_length
        us(:,t) = strejc_controller{xs(:,t)};
        xs(:,t+1) = A_s * xs(:,t) + B_s * us(:,t);
        ys(:,t+1) = C_s * xs(:,t+1);
    end

    ys_desc = ys * x_std + x_mean;
    us_desc = us * u_std + u_mean;
    e_s = ys_desc(1:end-1);%y

    u_sum_strejc = sum(abs(us_desc));
    y_sum_strejc = sum(abs(e_s));
    e_sum_strejc_y =sum(abs(e_s-x_mean));
    e_sum_strejc_u = sum(abs(us_desc-u_mean));
    rmse_strejc = sqrt(mean(e_s.^2));
    obj_strejc = sum(Qy * (e_s).^2 + Qu * (us_desc).^2);

    strejc_metrics = [u_sum_strejc, y_sum_strejc,e_sum_strejc_y, e_sum_strejc_u, rmse_strejc, obj_strejc];
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



save('results_20x0_MPC.mat', 'results_koopman', 'results_strejc', 'y0_vals');


%% --- Visualization of Metrics - updated (lines only, last 4 metrics) ---

metric_names = {'Sum |e| (y)', 'Sum |e| (u)', 'RMSE', 'Objective'};

% First 10 (lower initial conditions)
figure('Name','First 10 initial conditions - Metrics (Line)');
for m = 1:4
    subplot(2,2,m)
    plot(y0_vals(1:10), results_koopman(1:10,m+2), 'm-o','LineWidth',2); hold on;
    plot(y0_vals(1:10), results_strejc(1:10,m+2), 'b-s','LineWidth',2);
    xlabel('Initial condition y₀ (°C)');
    ylabel(metric_names{m});
    title(['Metric: ', metric_names{m}]);
    legend('Koopman','Strejc','Location','Best');
    grid on;grid minor;
end
sgtitle('Koopman vs Strejc - Metrics for First 10 Initial Conditions');

% Last 10 (higher initial conditions)
figure('Name','Last 10 initial conditions - Metrics (Line)');
for m = 1:4
    subplot(2,2,m)
    plot(y0_vals(11:20), results_koopman(11:20,m+2), 'm-o','LineWidth',2); hold on;
    plot(y0_vals(11:20), results_strejc(11:20,m+2), 'b-s','LineWidth',2);
    xlabel('Initial condition y₀ (°C)');
    ylabel(metric_names{m});
    title(['Metric: ', metric_names{m}]);
    legend('Koopman','Strejc','Location','Best');
    grid on;grid minor;
end
sgtitle('Koopman vs Strejc - Metrics for Last 10 Initial Conditions');


%% summary of metrics
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
