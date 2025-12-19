clc; clear;

% --- Python init ---
pyenv('Version','C:\Users\ivadu\AppData\Local\Programs\Python\Python39\python.exe');
py.sys.path().append('C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\ws2025\NewALL');
py.baseline_inference.init();

% --- load inputs ---
load('u_koopman.mat','uK');
load('u_strejc.mat','uS');

T = min(length(uK), length(uS));

yK = zeros(T+1,1);
yS = zeros(T+1,1);

% === Koopman baseline run ===
py.baseline_inference.get_x( yK(1) );   % init state
for k = 1:T
    y_next = py.baseline_inference.y_plus(uK(k));
    yK(k+1) = double(y_next);
end

% === RESET baseline ===
py.baseline_inference.init();
py.baseline_inference.get_x( yS(1) );

% === Strejc baseline run ===
for k = 1:T
    y_next = py.baseline_inference.y_plus(uS(k));
    yS(k+1) = double(y_next);
end

save('baseline_two_runs.mat','yK','yS');

load('baseline_two_runs.mat')

t = 0:length(yK)-1;

figure;
plot(t,yK,'m','LineWidth',2); hold on;
plot(t,yS,'b--','LineWidth',2);
legend('Koopman MPC','Strejc MPC');
ylabel('Outlet temperature (°C)');
xlabel('Time step');
grid on;
