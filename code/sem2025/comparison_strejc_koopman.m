clc; clear; close all;
%%
% Load results
load('results_koopman.mat', 'y_cl_desc', 'u_cl_desc','x_mean','u_mean');
y_koop = y_cl_desc(:);
u_koop = u_cl_desc(:);
koopman_last_u = u_cl_desc(end)
x_mean=x_mean(:);
x_std = 8.2880;

u_std = 26.3513;


load('results_strejc_to_zero.mat', 'y_cl_desc', 'u_cl_desc');
y_strejc = y_cl_desc(:);
u_strejc = u_cl_desc(:);
strejc_last_u = u_cl_desc(end)

sim_length = length(y_koop) - 1;
time = 0:sim_length;

% Control effort (sum of absolute values)
effort_koop = sum(abs(u_koop))
effort_strejc = sum(abs(u_strejc))

% Output magnitude
y_sum_koop = sum(abs(y_koop));
y_sum_strejc = sum(abs(y_strejc));

% Optional: error to 0°C
e_koop = y_koop -x_mean;  % reference = 0
e_strejc = y_strejc - x_mean;

e_sum_koop = sum(abs(e_koop))
e_sum_strejc = sum(abs(e_strejc))

e_koop_u = u_koop -u_mean;  % reference = 0
e_strejc_u = u_strejc - u_mean;

e_sum_koop_u = sum(abs(e_koop_u))
e_sum_strejc_u = sum(abs(e_strejc_u))

rmse_koop = sqrt(mean((y_koop-x_mean).^2));
rmse_strejc = sqrt(mean((y_strejc-x_mean).^2));

%% --- Metrics: Koopman vs Strejc ---

% Referenčná hodnota (cieľ výstupu = x_mean)
ref_y = x_mean;

% === RMSE už existuje ako: rmse_strejc, rmse_koop ===

% === Objective function (už počítaná)
% obj_strejc, obj_koop

% === Settling time (±0.5 °C)
tol = 0.02*ref_y;

% --- Settling time Strejc
y_err = abs(y_strejc - ref_y);
idx_settle = find(y_err > tol, 1, 'last');
settle_time_strejc = idx_settle + 1;  % index po ktorom už neprekročí

% --- Settling time Koopman
y_err = abs(y_koop - ref_y);
idx_settle = find(y_err > tol, 1, 'last');
settle_time_koop = idx_settle + 1;

%--- Display
fprintf('\n--- Closed-loop METRICS ---\n');
fprintf('RMSE              [°C]   → Strejc: %.2f\t Koopman: %.2f\n', rmse_strejc, rmse_koop);
fprintf('Control effort    [u²]   → Strejc: %.2f\t Koopman: %.2f\n', effort_strejc, effort_koop);
fprintf('Settling time     [steps]→ Strejc: %d\t Koopman: %d\n', settle_time_strejc, settle_time_koop);

%%
%obj value
Qy = 10;
Qu = 1;

obj_strejc=0;
obj_koop=0;
y_koop_scaled = (y_koop - x_mean) / x_std;
u_koop_scaled = (u_koop - u_mean) / u_std;
y_strej_scaled = (y_strejc - x_mean) / x_std;
u_strejc_scaled = (u_strejc - u_mean) / u_std;
for i=1:sim_length
    obj_koop = obj_koop + Qy*(y_koop_scaled(i))^2 + Qu*u_koop_scaled(i)^2; 
    obj_strejc= obj_strejc + Qy*(y_strej_scaled(i))^2 + Qu*u_strejc_scaled(i)^2; 
end



% Print comparison
fprintf('\n--- MPC Comparison ---\n');
fprintf('Sum |u|     Koopman: %.2f \t Strejc: %.2f\n', effort_koop, effort_strejc);
%fprintf('Sum |y|     Koopman: %.2f \t Strejc: %.2f\n', y_sum_koop, y_sum_strejc);
fprintf('Sum |e|     Koopman: %.2f \t Strejc: %.2f\n', e_sum_koop, e_sum_strejc);

fprintf('Sum |e_u |     Koopman: %.2f \t Strejc: %.2f\n', e_sum_koop_u, e_sum_strejc_u);
fprintf('Objective func    [unit] → Strejc: %.2f\t Koopman: %.2f\n', obj_strejc, obj_koop);

% 
% fprintf('RMSE (MPC Strejc vs 60 °C)   = %.4f\n', rmse_strejc);
% fprintf('RMSE (MPC Koopman vs 60 °C)  = %.4f\n', rmse_koop);
% 
time = 0:length(y_strejc)-1;

figure;
subplot(2,1,1)
settle_time_strejc = stepinfo(y_strejc, time, ref_y,'SettlingTimeThreshold', 0.02).SettlingTime;
settle_time_koop   = stepinfo(y_koop,   time, ref_y,'SettlingTimeThreshold', 0.02).SettlingTime;

plot(time, y_strejc , 'b-', 'LineWidth', 2); hold on;
plot(time, y_koop, 'm--', 'LineWidth', 2);
% xline(settle_time_koop, '--m');
% Strejc
% xline(settle_time_strejc, '--b');
xlabel('Time (s)'); ylabel('Output y (°C)');
legend('MPC Strejc', 'MPC Koopman','Location','best');  % Use LaTeX interpreter


title('Closed-loop Output');
grid on;box on; grid minor;

% ylim([45 65])
% === Označ settling time do grafu ===
subplot(2,1,2)
stairs(time(1:end-1), u_strejc, 'b-', 'LineWidth', 2); hold on;
stairs(time(1:end-1), u_koop, 'm--', 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Input u (%)');
legend('MPC Strejc', 'MPC Koopman','location','best');
title('Control Input');
grid on;box on;
saveas(gcf, 'C:\Users\ivadu\Desktop\8.semestrik\vymennik\prez\closed_loop_comparison50.png');


%% --- Open-loop Identification Comparison ---
% Load Strejc data
load('strejc_open_loop_comparison_data.mat', ...
    'y_open_desc', 'y_true', 'time', 'u_open_desc');

y_true = y_true(:);
y_open_strejc = y_open_desc(:);

y_true_scaled = (y_true - x_mean) / x_std;

% Load Koopman data
load('koopman_open_loop_comparison.mat', ...
    'y_koopman_desc');

y_open_koopman = y_koopman_desc(:);

y_open_strejc = y_open_strejc(2:end);
y_open_koopman = y_open_koopman(2:end);
y_open_desc = y_open_desc(1:end-1);
time = time(1:end-1);

% Compute RMSE (against the same y_true)
rmse_open_strejc = sqrt(mean((y_open_strejc - y_true).^2));
rmse_open_koopman = sqrt(mean((y_open_koopman - y_true).^2));
% rmse_koop_scaled = sqrt(mean((y_koop_scaled-y_true_scaled).^2))
% rmse_strej_scaled = sqrt(mean((y_strej_scaled-y_true_scaled).^2))

% Display RMSE
fprintf('\n--- Open-loop Prediction RMSE ---\n');
fprintf('RMSE (Open-loop Strejc)   = %.4f °C\n', rmse_open_strejc);
fprintf('RMSE (Open-loop Koopman)  = %.4f °C\n', rmse_open_koopman);

% Plot Open-loop Output Comparison
figure('Position', [100, 100, 600, 500]);  % výška a šírka upravená
t = tiledlayout(2,1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile
plot(time, y_true, 'k:', 'LineWidth', 1.5); hold on;
plot(time, y_open_strejc, 'b-', 'LineWidth', 2);
plot(time, y_open_koopman, 'm--', 'LineWidth', 2);
xlabel('Time (s)'); 
ylabel('Output y ($^\circ\mathrm{C}$)', 'Interpreter', 'latex');
lgd = legend('True Output', 'Strejc (RMSE = 3.55°C)', 'Koopman (RMSE = 2.44°C)', 'Location', 'southoutside','Orientation', 'horizontal');
lgd.FontSize = 10;  % or any smaller size you prefer
set(gca, 'FontName', 'Times New Roman');  % or your preferred font
lgd.FontName = 'Times New Roman';         % same font for legend

title('Prediction Using Testing Data');
grid on;

% Annotate RMSE values
rmse_text = sprintf('RMSE Strejc = %.2f°C\nRMSE Koopman = %.2f°C', rmse_open_strejc, rmse_open_koopman);
x_pos = time(end) * 0.55;  % place text toward the middle
y_pos = max([y_strejc; y_koop]) - 2;  % a bit below the top

% Create RMSE text outside plot using annotation box


nexttile
stairs(time, u_open_desc, 'b', 'LineWidth', 2); hold on
xlabel('Time (s)'); ylabel('Input u (\%)');
title('True Inputs');
    grid on;

saveas(gcf, 'C:\Users\ivadu\Desktop\8.semestrik\vymennik\prez\identification_comparison.png');

