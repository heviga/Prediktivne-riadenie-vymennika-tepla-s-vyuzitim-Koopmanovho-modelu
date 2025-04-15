close all
load('results_strejc.mat');
y_mpc_desc_strejc = y_mpc_desc;
u_mpc_desc_strejc = u_mpc_desc;

load('results_koopman.mat');
y_mpc_desc_koopman = y_cl_desc;
u_mpc_desc_koopman = u_cl_desc;



figure;
subplot(2,1,1)
plot(time, y_mpc_desc_strejc, 'b-', 'LineWidth', 2); hold on;
plot(time, y_mpc_desc_koopman, 'm--', 'LineWidth', 2);
xlabel('Time step'); ylabel('Output y (°C)');
legend('MPC Strejc', 'MPC Koopman');
title('Closed-loop Output: Strejc vs Koopman');
grid on;
ylim([40 80])

subplot(2,1,2)
stairs(time(1:end-1), u_mpc_desc_strejc, 'b-', 'LineWidth', 2); hold on;
stairs(time(1:end-1), u_mpc_desc_koopman, 'm--', 'LineWidth', 2);
xlabel('Time step'); ylabel('Input u');
legend('MPC Strejc', 'MPC Koopman');
title('Control Input Comparison');
grid on;
ylim([40 80])

ref = zeros(size(y_mpc_desc_strejc));  % 0 °C target
rmse_strejc = sqrt(mean((y_mpc_desc_strejc - ref).^2));
rmse_koopman = sqrt(mean((y_mpc_desc_koopman - ref).^2));

fprintf('RMSE (MPC Strejc vs 60 °C)   = %.4f\n', rmse_strejc);
fprintf('RMSE (MPC Koopman vs 60 °C)  = %.4f\n', rmse_koopman);

