
 %vacsie Qy -> high energy cost, fster contol, follow reference, 
 %vacsie Qu -> low energy cost, slower response
%save energy -> low Qy + high Qu
clc; clear; close all

%% Load models
A_k = double(readNPY('data/A_wC_all.npy'));
B_k = double(readNPY('data/B_wC_all.npy'));
C_k = double(readNPY('data/C_wC_all.npy'));
A_s = 0.97701252;
B_s = 0.03150894;
C_s = 1;

%% Load scaling data
load('train_data_ident.mat');
load('test_data_ident.mat');
Yall = [Ytrain(:); Ytest(:)];
Uall = [Utrain(:); Utest(:)];

x_mean = mean(Yall);
x_std = std(Yall);
u_mean = mean(Uall);
u_std = std(Uall);

%% Simulation parameters
N = 40;
sim_length = 200;
umin = (20 - u_mean) / u_std;
umax = (100 - u_mean) / u_std;
ymin = (0 - x_mean) / x_std;
ymax = (70 - x_mean) / x_std;

y0_val = 60; % Initial condition in °C
y0_scaled = (y0_val - x_mean) / x_std;
x0_k = pinv(C_k) * y0_scaled;
x0_s = y0_scaled;

%% Define Qy and Qu ranges
Qy_vals = [1, 10, 100];
Qu_vals = [0.1, 1, 10];

%% Run experiments
for Qy = Qy_vals
    for Qu = Qu_vals

        fprintf('\n---- Qy = %.2f | Qu = %.2f ----\n', Qy, Qu)

        % ----- Koopman -----
        u_k = sdpvar(repmat(1,1,N), repmat(1,1,N));
        x_k = sdpvar(repmat(size(A_k,1),1,N+1), repmat(1,1,N+1));
        x0_param = sdpvar(size(A_k,1),1);

        constr = [x_k{1} == x0_param];
        obj = 0;
        for k = 1:N
            constr = [constr, x_k{k+1} == A_k*x_k{k} + B_k*u_k{k}];
            yk = C_k * x_k{k};
            constr = [constr, umin <= u_k{k} <= umax];
            constr = [constr, ymin <= yk <= ymax];
            obj = obj + Qy*(yk)^2 + Qu*u_k{k}^2;
        end
        koop_ctrl = optimizer(constr, obj, sdpsettings('solver','quadprog'), x0_param, u_k{1});

        % Koopman simulation
        xk = zeros(size(A_k,1), sim_length+1);
        yk = zeros(1, sim_length+1);
        uk = zeros(1, sim_length);

        xk(:,1) = x0_k;
        for t = 1:sim_length
            uk(:,t) = koop_ctrl{xk(:,t)};
            xk(:,t+1) = A_k * xk(:,t) + B_k * uk(:,t);
            yk(:,t+1) = C_k * xk(:,t+1);
        end

        yk_desc = yk * x_std + x_mean;
        uk_desc = uk * u_std + u_mean;
        e_k = yk_desc(1:end-1);

        fprintf('Koopman | Sum |u|: %.2f | Sum |y|: %.2f | Obj: %.2f\n', sum(abs(uk_desc)), sum(abs(e_k)), sum(Qy*(e_k).^2 + Qu*(uk_desc).^2))

        % ----- Strejc -----
        u_s = sdpvar(repmat(1,1,N), repmat(1,1,N));
        x_s = sdpvar(repmat(1,1,N+1), repmat(1,1,N+1));
        x0_param = sdpvar(1,1);

        constr = [x_s{1} == x0_param];
        obj = 0;
        for k = 1:N
            constr = [constr, x_s{k+1} == A_s*x_s{k} + B_s*u_s{k}];
            ys = C_s * x_s{k};
            constr = [constr, umin <= u_s{k} <= umax];
            constr = [constr, ymin <= ys <= ymax];
            obj = obj + Qy*(ys)^2 + Qu*u_s{k}^2;
        end
        strejc_ctrl = optimizer(constr, obj, sdpsettings('solver','quadprog'), x0_param, u_s{1});

        % Strejc simulation
        xs = zeros(1, sim_length+1);
        ys = zeros(1, sim_length+1);
        us = zeros(1, sim_length);

        xs(:,1) = x0_s;
        for t = 1:sim_length
            us(:,t) = strejc_ctrl{xs(:,t)};
            xs(:,t+1) = A_s * xs(:,t) + B_s * us(:,t);
            ys(:,t+1) = C_s * xs(:,t+1);
        end

        ys_desc = ys * x_std + x_mean;
        us_desc = us * u_std + u_mean;
        e_s = ys_desc(1:end-1);

        fprintf('Strejc  | Sum |u|: %.2f | Sum |y|: %.2f | Obj: %.2f\n', sum(abs(us_desc)), sum(abs(e_s)), sum(Qy*(e_s).^2 + Qu*(us_desc).^2))
    end
end


%control effort - Sum |u| is similar in all cases
