%clc; 
function u_koop = mpc_koop(x_koop,r)

% Add path for readNPY function (automatically added by startup.m, but ensure it's available)
if exist('readNPY', 'file') == 0
    addpath('../..');  % Add code directory to path
end

% Load data
load('train_data.mat');  % Ytrain, Utrain
load('test_data.mat');   % Ytest, Utest

A = double(readNPY('data/A_wC.npy'));
B = double(readNPY('data/B_wC.npy'));
C = double(readNPY('data/C_wC.npy'));
D = 0;

% Flatten
Ytrain = Ytrain(:);
Utrain = Utrain(:);
Ytest = Ytest(:);
Utest = Utest(:);

% Full dataset for scaling
Yall = [Ytrain; Ytest];
Uall = [Utrain; Utest];

x_mean = mean(Yall);
x_std = std(Yall);
u_mean = mean(Uall);
u_std = std(Uall);

% Get dimensions from loaded matrices
nx = size(A, 1);  % Koopman state dimension (10)
ny = size(C, 1);  % Output dimension (1)
nu = size(B, 2);  % Input dimension (1)
Ts = 1;
% sim_length = 300;

% Scale initial condition
%x0 = (50 - x_mean) / x_std;   % Start from 50°C

%% --- MPC setup for control to 0°C ---
Qy = 10;
Qu = 1;
N = 40;

%r = (0 - x_mean) / x_std;   % Setpoint = 0°C (scaled)
umin = (20 - u_mean) / u_std;
umax = (100 - u_mean) / u_std;
ymin = (0 - x_mean) / x_std;
ymax = (70 - x_mean) / x_std;

% YALMIP vars
u = sdpvar(repmat(1,1,N), repmat(1,1,N));
x = sdpvar(repmat(nx,1,N+1), repmat(1,1,N+1));
x0_param = sdpvar(nx,1);

constraints = []; objective = 0;
constraints = [constraints; x{1} == x0_param];
% for k = 1:N
%     constraints = [constraints, x{k+1} == A*x{k} + B*u{k}];
%     constraints = [constraints, umin <= u{k} <= umax];
%     yk = C*x{k};
%     constraints = [constraints, ymin <= yk <= ymax];
%     objective = objective + Qy*(yk)^2 + Qu*u{k}^2; % r prec ->riadenie do nuly
% end


for k = 1:N
    constraints = [constraints; x{k+1} == A*x{k} + B*u{k}];
    constraints = [constraints; umin <= u{k} <= umax];%na zasahy
    yk = C * x{k};
    constraints = [constraints; ymin <= yk <= ymax];
    objective = objective + Qy*(yk-r)^2 + Qu*u{k}^2;  % Control to 0
end

controller = optimizer(constraints, objective, sdpsettings('solver','quadprog'), x0_param, u{1});

% Compute optimal control input
u_koop = controller{x_koop(:,1)};

end
%mpc_koop([u(1);u(2)],[u(3)]) 

Q_KF = 0.5*eye(10);   % process noise cov
R_KF = 0.1;   % measurement noise cov
P = eye(nx);   % initial covariance
meas_noise_std = 0;%sqrt(R_KF);



