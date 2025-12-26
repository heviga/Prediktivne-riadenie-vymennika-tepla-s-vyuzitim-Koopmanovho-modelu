clc; clear; 
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');

%% ===== Load results =====
% Check which files exist
if exist('results_koopman_to_zero.mat', 'file')
    K = load('results_koopman_to_zero.mat');
    fprintf('Loaded: results_koopman_to_zero.mat\n');
elseif exist('results_koopman.mat', 'file')
    K = load('results_koopman.mat');
    fprintf('Loaded: results_koopman.mat\n');
else
    error('Koopman results file not found! Run cl_koopman first.');
end

yK_true = K.y_true_desc(:);    % Baseline output with Koopman MPC inputs
yK_est = K.y_est_desc(:);       % Kalman filter estimate (Koopman model)
uK = K.u_cl_desc(:);            % Koopman MPC control inputs

% Check which Strejc file exists
if exist('results_strejc_to_zero.mat', 'file')
    S = load('results_strejc_to_zero.mat');
    fprintf('Loaded: results_strejc_to_zero.mat\n');
elseif exist('results_strejc_loaded_baseline.mat', 'file')
    S = load('results_strejc_loaded_baseline.mat');
    fprintf('Loaded: results_strejc_loaded_baseline.mat (old file)\n');
    warning('Using old Strejc results file. Run cl_strejc to generate new results.');
else
    error('Strejc results file not found! Run cl_strejc first.');
end

yS_true = S.y_true_desc(:);     % Baseline output with Strejc MPC inputs
yS_est = S.y_est_desc(:);       % Kalman filter estimate (Strejc model)
uS = S.u_cl_desc(:);            % Strejc MPC control inputs

%% ===== Extract signals =====
% NOTE: yK_true and yS_true should be DIFFERENT if uK and uS are different
% because they represent baseline model outputs with different control inputs
fprintf('Checking if inputs are different...\n');
input_diff = norm(uK(1:min(length(uK),length(uS))) - uS(1:min(length(uK),length(uS))));
fprintf('Input difference (norm): %.4f\n', input_diff);

fprintf('Checking if baseline outputs are different...\n');
output_diff = norm(yK_true(1:min(length(yK_true),length(yS_true))) - yS_true(1:min(length(yK_true),length(yS_true))));
fprintf('Baseline output difference (norm): %.4f\n', output_diff);

if output_diff < 1e-6 && input_diff > 1e-6
    warning('WARNING: Inputs are different but baseline outputs are the same!');
    warning('This suggests the baseline model may not be responding correctly to different inputs.');
    warning('Possible causes:');
    warning('1. Baseline model state not reset between simulations');
    warning('2. Input scaling mismatch between cl_koopman and cl_strejc');
    warning('3. Baseline model using cached/incorrect state');
end



%% ===== Align lengths =====
T = min([length(yK_true), length(yS_true), length(yK_est), length(yS_est)]);
yK_true = yK_true(1:T);
yS_true = yS_true(1:T);
yK_est  = yK_est(1:T);
yS_est  = yS_est(1:T);

Tu = min([length(uK), length(uS), T-1]);
uK = uK(1:Tu);
uS = uS(1:Tu);

%% ===== Reference =====
y_ref = K.x_mean;   % regulation target

%% ===== Metrics (optional print) =====
% RMSE of estimates vs reference
rmseK_est = sqrt(mean((yK_est - y_ref).^2));
rmseS_est = sqrt(mean((yS_est - y_ref).^2));

% RMSE of baseline outputs vs reference
rmseK_true = sqrt(mean((yK_true - y_ref).^2));
rmseS_true = sqrt(mean((yS_true - y_ref).^2));

fprintf('\n=== Performance Metrics ===\n');
fprintf('Koopman MPC - KF Estimate RMSE: %.3f °C\n', rmseK_est);
fprintf('Strejc MPC  - KF Estimate RMSE: %.3f °C\n', rmseS_est);
fprintf('Koopman MPC - Baseline Output RMSE: %.3f °C\n', rmseK_true);
fprintf('Strejc MPC  - Baseline Output RMSE: %.3f °C\n', rmseS_true);
fprintf('==========================\n\n');

%% ===== Time vectors =====
tY = 0:T-1;
tU = 0:Tu-1;

%% ===== Plot =====
figure('Position',[100 100 900 520]);

% === OUTPUT ===
subplot(2,1,1)
plot(tY, yK_est, 'm','LineWidth',2.2); hold on;
plot(tY, yS_est, 'b--','LineWidth',2.2);
yline(y_ref,'k-','LineWidth',1.4);

ylabel('Outlet temperature ($^\circ$C)');
title('Closed-loop response');
legend('Koopman MPC','Strejc MPC','Reference','Location','best');
grid on; grid minor;
ylim([40 70])


% === INPUT (control) ===
subplot(2,1,2)
stairs(tU, uK, 'm','LineWidth',2); hold on;
stairs(tU, uS, 'b--','LineWidth',2);

xlabel('Time step');
ylabel('Pump speed (\%)');
title('Control input');
legend('Koopman MPC','Strejc MPC','Location','best');
grid on; grid minor;

%% ===== Save =====
% Create figs directory if it doesn't exist
if ~exist('figs', 'dir')
    mkdir('figs');
end
saveas(gcf,'figs/compare_cl_temperature_input.png');
fprintf('Figure saved to: figs/compare_cl_temperature_input.png\n');
