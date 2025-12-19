clc; clear; close all;

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');

%% ===== Load results =====
K = load('results_koopman_to_zero.mat');          % y_true_desc, y_est_desc, u_cl_desc, x_mean
yK = K.y_est_desc(:);          % Koopman MPC regulated output
uK = K.u_cl_desc(:);

S = load('results_strejc_loaded_baseline.mat');   % y_true_desc, y_est_desc, u_cl_desc
yS = S.y_est_desc(:);          % Strejc MPC regulated output
uS = S.u_cl_desc(:);

%% ===== Extract signals =====

y_base = K.y_true_desc(:);     % baseline plant (common)



%% ===== Align lengths =====
T = min([length(y_base), length(yK), length(yS)]);
y_base = y_base(1:T);
yK     = yK(1:T);
yS     = yS(1:T);

Tu = min([length(uK), length(uS), T-1]);
uK = uK(1:Tu);
uS = uS(1:Tu);

%% ===== Reference =====
y_ref = K.x_mean;   % regulation target

%% ===== Metrics (optional print) =====
rmseK = sqrt(mean((yK - y_ref).^2));
rmseS = sqrt(mean((yS - y_ref).^2));

fprintf('Closed-loop RMSE Koopman: %.3f °C\n', rmseK);
fprintf('Closed-loop RMSE Strejc : %.3f °C\n', rmseS);

%% ===== Time vectors =====
tY = 0:T-1;
tU = 0:Tu-1;

%% ===== Plot =====
figure('Position',[100 100 900 520]);

% === OUTPUT ===
subplot(2,1,1)
plot(tY, K.y_est_desc(1:T), 'm','LineWidth',2.2); hold on;
plot(tY, S.y_est_desc(1:T), 'b--','LineWidth',2.2);
yline(y_ref,'k-','LineWidth',1.4);

ylabel('Outlet temperature ($^\circ$C)');
title('Estimated closed-loop response');
legend('Koopman MPC (estimate)','Strejc MPC (estimate)','Reference');
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
saveas(gcf,'figs/compare_cl_temperature_input.png');
