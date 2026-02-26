%% cl_multiIC_using_control_functions.m
% Closed-loop simulation using REAL control functions
% Plant = baseline_inference (scaled domain internally)

clc; clear; close all;

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');

%% ===== SETTINGS =====
temps      = [45, 50, 55, 58, 60, 62, 66, 68];
sim_length = 300;

Qy = 10;
Qu = 1;

bias = 0.3160;   % baseline correction (NECHÁVAME)

%% ===== LOAD SCALING =====
load('train_data.mat');
load('test_data.mat');

Yall = [Ytrain(:); Ytest(:)];
Uall = [Utrain(:); Utest(:)];

x_mean = mean(Yall);
x_std  = std(Yall);
u_mean = mean(Uall);
u_std  = std(Uall);

%% ===== PYTHON BASELINE SETUP =====
pyenv('Version','C:\Users\ivadu\AppData\Local\Programs\Python\Python39\python.exe');
py.sys.path().append('C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\LS2026');


%% ===== STORAGE =====
metrics_rows = table('Size',[0 7], ...
    'VariableTypes',{'double','double','double','double','double','double','double'}, ...
    'VariableNames',{'T0','RMSE_K','RMSE_S','IAE_K','IAE_S','Obj_K','Obj_S'});

%% ===== MAIN LOOP =====
for i = 1:length(temps)

    T0 = temps(i);
    fprintf('\n=== IC %d/%d: T0 = %d °C ===\n',i,length(temps),T0);

    %% RESET
    py.baseline_inference.init();
    control_koopman(0, struct('reset',true));
    control_strejc(0, struct('reset',true));

    %% INITIAL CONDITIONS
    yK = T0;
    yS = T0;

    y_store_K = zeros(sim_length+1,1);
    y_store_S = zeros(sim_length+1,1);
    u_store_K = zeros(sim_length,1);
    u_store_S = zeros(sim_length,1);

    y_store_K(1) = yK;
    y_store_S(1) = yS;

    %% ===== KOOPMAN =====
    py.baseline_inference.init();
    py.baseline_inference.get_x((yK - x_mean)/x_std);

    for t = 1:sim_length

        u = control_koopman(yK + bias);
        u_store_K(t) = u;

        u_scaled = (u - u_mean)/u_std;
        y_next_scaled = py.baseline_inference.y_plus(u_scaled);

        yK = double(y_next_scaled.item())*x_std + x_mean;
        y_store_K(t+1) = yK;
    end

    %% ===== STREJC =====
    py.baseline_inference.init();
    py.baseline_inference.get_x((yS - x_mean)/x_std);

    for t = 1:sim_length

        u = control_strejc(yS + bias);
        u_store_S(t) = u;

        u_scaled = (u - u_mean)/u_std;
        y_next_scaled = py.baseline_inference.y_plus(u_scaled);

        yS = double(y_next_scaled.item())*x_std + x_mean;
        y_store_S(t+1) = yS;
    end

    %% ===== METRICS =====
    y_ref = x_mean;

    % errors (physical)
    eK = y_store_K(1:end-1) - y_ref;
    eS = y_store_S(1:end-1) - y_ref;

    rmseK = sqrt(mean(eK.^2));
    rmseS = sqrt(mean(eS.^2));

    iaeK = sum(abs(eK));
    iaeS = sum(abs(eS));

    % scaled signals for objective
    y_scaled_K = (y_store_K(1:end-1) - x_mean)/x_std;
    y_scaled_S = (y_store_S(1:end-1) - x_mean)/x_std;

    u_scaled_K = (u_store_K - u_mean)/u_std;
    u_scaled_S = (u_store_S - u_mean)/u_std;

    objK = sum(Qy*y_scaled_K.^2 + Qu*u_scaled_K.^2);
    objS = sum(Qy*y_scaled_S.^2 + Qu*u_scaled_S.^2);

    % append row
    metrics_rows = [metrics_rows;
        {T0, rmseK, rmseS, iaeK, iaeS, objK, objS}];

    fprintf('Koop RMSE=%.3f | Strejc RMSE=%.3f\n',rmseK,rmseS);

%% ===== PER-IC PLOT =====
tY = 0:sim_length;
tU = 0:sim_length-1;

figure('Color','w','Position',[100 100 900 520]);
tiledlayout(2,1,'TileSpacing','Compact','Padding','Compact');

% OUTPUT
nexttile;
plot(tY,y_store_K,'m','LineWidth',2); hold on;
plot(tY,y_store_S,'b--','LineWidth',2);
yline(x_mean,'k','LineWidth',1.2);
grid on; grid minor;
ylabel('Outlet temperature ($^\circ$C)');
title(sprintf('Closed-loop (start %d$^\\circ$C)',T0));
legend('Koopman MPC','Strejc MPC','Steady-state','Location','best');
ylim auto   % tvoje thesis limity
xlim([0 150]);

% INPUT
nexttile;
plot(tU,u_store_K,'m','LineWidth',2); hold on;
plot(tU,u_store_S,'b--','LineWidth',2);
grid on; grid minor;
xlabel('Time step');
ylabel('Pump speed (\%)');
legend('Koopman MPC','Strejc MPC','Location','best');
ylim auto;
xlim([0 150]);
end

%% ===== SUMMARY =====
disp(metrics_rows);

%% ===== SUMMARY PLOT =====
T0 = metrics_rows.T0;

figure('Color','w','Position',[100 100 1050 360]);
tiledlayout(1,3,'TileSpacing','Compact','Padding','Compact');

nexttile;
plot(T0, metrics_rows.RMSE_K,'m-o','LineWidth',1.6); hold on;
plot(T0, metrics_rows.RMSE_S,'b-s','LineWidth',1.6);
grid on; title('RMSE');

nexttile;
plot(T0, metrics_rows.IAE_K,'m-o','LineWidth',1.6); hold on;
plot(T0, metrics_rows.IAE_S,'b-s','LineWidth',1.6);
grid on; title('IAE');

nexttile;
plot(T0, metrics_rows.Obj_K,'m-o','LineWidth',1.6); hold on;
plot(T0, metrics_rows.Obj_S,'b-s','LineWidth',1.6);
grid on; title('Closed-loop Objective');