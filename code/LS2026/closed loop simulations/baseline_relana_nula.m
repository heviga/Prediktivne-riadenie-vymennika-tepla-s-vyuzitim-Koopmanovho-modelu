%% ===== BASELINE TRUE STEADY STATE IDENTIFICATION ===== 61.688441

u_mean_phys = 65.8447;   % tvoje fyzické u_mean
u_mean_scaled = (u_mean_phys - u_mean)/u_std;

sim_ss = 250;            % 600 krokov
y_ss = zeros(sim_ss,1);

% inicializuj baseline v nule (scaled)
py.baseline_inference.get_x(0);

y_curr = 0;

for k = 1:sim_ss
    y_next = py.baseline_inference.y_plus(u_mean_scaled);
    y_curr = double(y_next.item());
    y_ss(k) = y_curr;
end

% priemer posledných 150 krokov
x_mean_baseline_scaled = mean(y_ss);
x_mean_baseline_phys   = x_mean_baseline_scaled * x_std + x_mean;

fprintf('\n===== BASELINE TRUE STEADY STATE =====\n');
fprintf('Scaled steady-state: %.6f\n', x_mean_baseline_scaled);
fprintf('Physical steady-state: %.6f °C\n\n', x_mean_baseline_phys);
%
%% ==========================================
%  BASELINE STEADY-STATE TEST (NO MPC)
% ==========================================

clc; clear; close all;

%% ===== PYTHON BASELINE INIT =====
pyenv('Version', 'C:\Users\ivadu\AppData\Local\Programs\Python\Python39\python.exe');
py.sys.path().append('C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\ws2025\NewALL');
py.baseline_inference.init();

%% ===== LOAD ORIGINAL SCALING =====
load('train_data.mat');
load('test_data.mat');

Yall = [Ytrain(:); Ytest(:)];
Uall = [Utrain(:); Utest(:)];

x_mean = mean(Yall);
x_std  = std(Yall);
u_mean = mean(Uall);
u_std  = std(Uall);

fprintf("Original scaling:\n");
fprintf("x_mean = %.6f\n", x_mean);
fprintf("x_std  = %.6f\n", x_std);
fprintf("u_mean = %.6f\n", u_mean);
fprintf("u_std  = %.6f\n\n", u_std);

%% ===== SIMULATION SETTINGS =====
sim_steps = 600;
avg_last  = 250;   % mean z posledných 150 krokov

%% ===== TEST INPUT (Fyzická hodnota) =====
u_const_phys = u_mean;   % steady input
u_const_scaled = (u_const_phys - u_mean) / u_std;

fprintf("Testing constant input: %.4f %%\n\n", u_const_phys);

%% ===== RESET BASELINE TO SCALED 0 =====
% scaled 0 znamená fyzicky x_mean
py.baseline_inference.get_x(0);

%% ===== RUN SIMULATION =====
y_scaled = zeros(sim_steps,1);

for k = 1:sim_steps
    y_next = py.baseline_inference.y_plus(u_const_scaled);
    y_scaled(k) = double(y_next.item());
end

%% ===== DESCALE OUTPUT =====
y_phys = y_scaled * x_std + x_mean;

%% ===== STEADY-STATE ESTIMATE =====
y_ss = mean(y_phys(end-avg_last+1:end));

fprintf("Baseline steady-state (mean last %d steps): %.6f °C\n", ...
        avg_last, y_ss);

fprintf("Difference vs x_mean: %.6f °C\n\n", ...
        y_ss - x_mean);

%% ===== PLOT =====
figure('Color','w');
plot(y_phys,'LineWidth',1.5);
yline(y_ss,'r--','LineWidth',1.5);
yline(x_mean,'k:','LineWidth',1.5);
grid on; grid minor;
xlabel('Time step');
ylabel('T4 (°C)');
title('Baseline steady-state test');
legend('Baseline output','Estimated steady-state','x_{mean}');


%%
sim_test = 400;
u_const_scaled = 0;

py.baseline_inference.get_x(0);

for k = 1:sim_test
    y_next = py.baseline_inference.y_plus(u_const_scaled);
    y_test(k) = double(y_next.item());
end

mean(y_test)
0.3160 * x_std