%clc; torch.save(best_model, "./data/model_baseline.pth")
clear all; %close all

% IMPORTANT: Clear Python interface first to avoid file lock issues
try
    clear py;
catch
    % py might not exist yet, that's fine
end

% Configure Python environment for baseline inference
python_exe = 'C:\Users\ivadu\AppData\Local\Programs\Python\Python39\python.exe';
python_path = 'C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\ws2025\NewALL';

% Install missing dependencies first (before initializing pyenv)
fprintf('Installing/upgrading Python dependencies...\n');
[status1, cmdout1] = system(sprintf('"%s" -m pip install --upgrade zipp setuptools 2>&1', python_exe));
if status1 == 0
    fprintf('Basic dependencies installed successfully.\n');
else
    fprintf('Warning: Basic dependency installation may have failed: %s\n', cmdout1);
end

% Check if mlflow 2.5.0 is already installed correctly
fprintf('Checking mlflow installation...\n');
[status_check, version_check] = system(sprintf('"%s" -m pip show mlflow 2>&1', python_exe));
mlflow_installed = contains(version_check, 'Version: 2.5.0');

if ~mlflow_installed
    % Completely uninstall mlflow to fix corrupted installation
    fprintf('Uninstalling mlflow to fix corrupted installation...\n');
    system(sprintf('"%s" -m pip uninstall -y mlflow mlflow-skinny 2>&1', python_exe));
    
    % Clear Python cache to avoid import issues
    fprintf('Clearing Python cache...\n');
    python_cache_dir = fullfile(python_path, '__pycache__');
    if exist(python_cache_dir, 'dir')
        rmdir(python_cache_dir, 's');
    end
    
    % Reinstall mlflow with the exact version required by neuromancer
    fprintf('Reinstalling mlflow with version required by neuromancer (2.5.0)...\n');
    [status2, cmdout2] = system(sprintf('"%s" -m pip install --no-cache-dir mlflow==2.5.0 2>&1', python_exe));
    if status2 ~= 0
        fprintf('Standard installation failed, trying with --user flag...\n');
        % Try with --user flag to avoid permission issues
        [status3, cmdout3] = system(sprintf('"%s" -m pip install --user --no-cache-dir mlflow==2.5.0 2>&1', python_exe));
        if status3 == 0
            fprintf('MLflow 2.5.0 installed successfully (user installation).\n');
        else
            fprintf('Warning: MLflow installation failed: %s\n', cmdout3);
        end
    else
        fprintf('MLflow 2.5.0 installed successfully.\n');
    end
else
    fprintf('MLflow 2.5.0 is already installed correctly.\n');
end

try
    % Set up Python environment
    pyenv('Version', python_exe);
    
    % Add Python path for baseline_inference
    if count(py.sys.path, python_path) == 0
        py.sys.path().append(python_path);
    end
    
    % Add user site-packages to path (in case --user installation was used)
    % Construct user site-packages path manually
    [py_dir, ~, ~] = fileparts(python_exe);
    user_site = fullfile(getenv('APPDATA'), 'Python', 'Python39', 'site-packages');
    if exist(user_site, 'dir') && count(py.sys.path, user_site) == 0
        py.sys.path().append(user_site);
        fprintf('Added user site-packages to Python path.\n');
    end
    
    % Clear Python bytecode cache to avoid circular import issues
    fprintf('Clearing Python bytecode cache...\n');
    try
        % Clear mlflow cache specifically
        [py_dir, ~, ~] = fileparts(python_exe);
        mlflow_cache = fullfile(py_dir, 'Lib', 'site-packages', 'mlflow', '__pycache__');
        if exist(mlflow_cache, 'dir')
            rmdir(mlflow_cache, 's');
        end
        % Also clear version.pyc if it exists
        version_pyc = fullfile(py_dir, 'Lib', 'site-packages', 'mlflow', 'version.pyc');
        if exist(version_pyc, 'file')
            delete(version_pyc);
        end
    catch
        fprintf('Could not clear mlflow cache, continuing...\n');
    end
    
    % Skip mlflow verification - let neuromancer handle the import
    % The circular import might be resolved when importing through neuromancer
    fprintf('Skipping direct mlflow import (will be handled by neuromancer)...\n');
    
    % Test if mlflow can be imported in a fresh Python subprocess
    fprintf('Testing mlflow import in fresh Python subprocess...\n');
    test_cmd = sprintf('"%s" -c "import mlflow; print(mlflow.__version__)" 2>&1', python_exe);
    [test_status, test_output] = system(test_cmd);
    if test_status == 0
        fprintf('MLflow imports successfully in Python subprocess.\n');
        fprintf('This suggests the issue is with MATLAB''s Python interface caching.\n');
    else
        fprintf('MLflow also fails in Python subprocess: %s\n', test_output);
        fprintf('The mlflow installation may be corrupted.\n');
    end
    
    % Import the module explicitly
    fprintf('Importing baseline_inference module...\n');
    try
        baseline_inference = py.importlib.import_module('baseline_inference');
    catch import_err
        % If import fails, provide helpful error message and workaround
        if contains(import_err.message, 'mlflow') || contains(import_err.message, 'version')
            fprintf('\n========================================\n');
            fprintf('CHYBA: MLFLOW CIRCULAR IMPORT ERROR\n');
            fprintf('========================================\n');
            fprintf('Toto je známy problém s mlflow 2.5.0 v MATLAB.\n\n');
            fprintf('RIEŠENIE - Skúste v tomto poradí:\n');
            fprintf('1. Spustite fix_mlflow.py skript:\n');
            fprintf('   python fix_mlflow.py\n');
            fprintf('   alebo\n');
            fprintf('   "%s" "%s"\n', python_exe, fullfile(python_path, 'fix_mlflow.py'));
            fprintf('\n2. Ak to nepomôže, zatvorte MATLAB úplne a znova ho otvorte\n');
            fprintf('3. Skúste znova spustiť cl_strejc\n');
            fprintf('========================================\n');
            error('Nepodarilo sa importovať baseline_inference kvôli mlflow circular import.\nSkúste riešenie vyššie.');
        else
            rethrow(import_err);
        end
    end
    
    % Initialize baseline inference
    baseline_inference.init();
    fprintf('Baseline inference initialized successfully.\n');
catch ME
    % If import fails, try to get more details and retry
    fprintf('First import attempt failed: %s\n', ME.message);
    fprintf('Attempting to fix mlflow circular import issue...\n');
    
    try
        % Clear Python interface completely
        clear py;
        
        % Completely remove mlflow and reinstall
        fprintf('Performing deep clean of mlflow installation...\n');
        system(sprintf('"%s" -m pip uninstall -y mlflow mlflow-skinny 2>&1', python_exe));
        
        % Clear site-packages cache for mlflow more thoroughly
        fprintf('Clearing mlflow cache directories...\n');
        [filepath, ~, ~] = fileparts(python_exe);
        mlflow_dir = fullfile(filepath, 'Lib', 'site-packages', 'mlflow');
        mlflow_cache = fullfile(mlflow_dir, '__pycache__');
        
        % Remove all .pyc files in mlflow directory
        if exist(mlflow_dir, 'dir')
            try
                % Remove __pycache__ directory
                if exist(mlflow_cache, 'dir')
                    rmdir(mlflow_cache, 's');
                end
                % Remove version.pyc specifically
                version_pyc = fullfile(mlflow_dir, 'version.pyc');
                if exist(version_pyc, 'file')
                    delete(version_pyc);
                end
                % Remove __init__.pyc
                init_pyc = fullfile(mlflow_dir, '__init__.pyc');
                if exist(init_pyc, 'file')
                    delete(init_pyc);
                end
            catch
                fprintf('Could not remove mlflow cache files, continuing...\n');
            end
        end
        
        % Reinstall mlflow with the exact version required by neuromancer
        fprintf('Reinstalling mlflow 2.5.0 (required by neuromancer)...\n');
        [status, ~] = system(sprintf('"%s" -m pip install --user --no-cache-dir --force-reinstall mlflow==2.5.0 2>&1', python_exe));
        if status ~= 0
            % Try without --user flag
            fprintf('Trying standard installation method...\n');
            [status2, ~] = system(sprintf('"%s" -m pip install --no-cache-dir --force-reinstall mlflow==2.5.0 2>&1', python_exe));
            if status2 ~= 0
                % Last resort: try without --force-reinstall
                fprintf('Trying installation without force-reinstall...\n');
                system(sprintf('"%s" -m pip install --user mlflow==2.5.0 2>&1', python_exe));
            end
        end
        
        % Try to fix mlflow circular import by patching __init__.py
        fprintf('Attempting to fix mlflow circular import issue...\n');
        fix_script = fullfile(python_path, 'fix_mlflow.py');
        if exist(fix_script, 'file')
            fprintf('Running mlflow fix script...\n');
            [fix_status, fix_output] = system(sprintf('"%s" "%s" 2>&1', python_exe, fix_script));
            fprintf('%s\n', fix_output);
            if fix_status == 0
                fprintf('MLflow fix script completed successfully.\n');
            else
                fprintf('MLflow fix script had issues, but continuing...\n');
            end
        else
            fprintf('Warning: fix_mlflow.py not found, skipping patch...\n');
        end
        
        % Wait a moment for file system to settle
        pause(2);
        
        % Completely terminate and restart Python environment
        try
            terminate(pyenv);
        catch
            % If terminate fails, just clear py
            clear py;
        end
        pause(1); % Wait for Python to fully terminate
        pyenv('Version', python_exe);
        
        if count(py.sys.path, python_path) == 0
            py.sys.path().append(python_path);
        end
        if exist(user_site, 'dir') && count(py.sys.path, user_site) == 0
            py.sys.path().append(user_site);
        end
        
        % Invalidate Python's import cache
        try
            importlib = py.importlib.import_module('importlib');
            importlib.invalidate_caches();
            fprintf('Python import cache invalidated.\n');
        catch
            fprintf('Could not invalidate import cache, continuing...\n');
        end
        
        % Try importing again
        baseline_inference = py.importlib.import_module('baseline_inference');
        baseline_inference.init();
        fprintf('Baseline inference initialized successfully after mlflow fix.\n');
    catch ME2
        fprintf('Retry failed: %s\n', ME2.message);
        fprintf('Attempting workaround: importing with importlib.reload...\n');
        
        try
            % Final attempt: clear and reload
            clear py;
            pyenv('Version', python_exe);
            
            if count(py.sys.path, python_path) == 0
                py.sys.path().append(python_path);
            end
            
            % Import sys and set environment variable to avoid mlflow issues
            py.sys.path().insert(int32(0), python_path);
            
            baseline_inference = py.importlib.import_module('baseline_inference');
            baseline_inference.init();
            fprintf('Baseline inference initialized successfully with workaround.\n');
        catch ME3
            error('Failed to initialize baseline inference after all attempts: %s', ME3.message);
        end
    end
end

% Load data
load('train_data.mat');  % Ytrain, Utrain
load('test_data.mat');   % Ytest, Utest

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

% Discrete Strejc model parameters


A = 0.98540172;
B = 0.01640409;
C = 1;
D = 0;

nx = 1;
ny = 1;
nu = 1;
Ts = 1;
sim_length = 300;

% Scale initial condition
x0 = (50 - x_mean) / x_std;   % Start from 50°C

%% --- MPC setup for control to 0°C ---
Qy = 10;
Qu = 1;
N = 20;

r = (0 - x_mean) / x_std;   % Setpoint = 0°C (scaled)
umin = (20 - u_mean) / u_std;
umax = (100 - u_mean) / u_std;
ymin = (0 - x_mean) / x_std;
ymax = (70 - x_mean) / x_std;

% YALMIP vars
u = sdpvar(repmat(1,1,N), repmat(1,1,N));
x = sdpvar(repmat(nx,1,N+1), repmat(1,1,N+1));
x0_param = sdpvar(nx,1);

constraints = [x{1} == x0_param];
objective = 0;

for k = 1:N
    constraints = [constraints, x{k+1} == A*x{k} + B*u{k}];
    constraints = [constraints, umin <= u{k} <= umax];%na zasahy
    yk = C * x{k};
    constraints = [constraints, ymin <= yk <= ymax];%na stavy
    objective = objective + Qy*(yk)^2 + Qu*u{k}^2;  % Control to 0
end

controller = optimizer(constraints, objective, sdpsettings('solver','quadprog'), x0_param, u{1});

%% --- Kalman filter parameters (scaled domain) ---
% Tieto hodnoty môžeš doladiť podľa reality:
Q_kalman = 0.5;    % process noise covariance (small: model veríme)
R_kalman = 0.1;    % measurement noise covariance (väčšie -> menej dôvery v meranie)
P0 = 1;             % počiatočná kovariancia
meas_noise_std = 0;%sqrt(R_kalman); % pre simulované merania
%% --- Simulate closed-loop ---
x_true = zeros(nx, sim_length+1);
y_true = zeros(ny, sim_length+1); %tvarime sa ze je to merane
u_cl = zeros(nu, sim_length);
x_est = zeros(nx, sim_length+1);   % Kalman estimated state history
y_meas = zeros(ny, sim_length+1); % pre sum
P=P0;

%initial true, estimated state
x_true(:,1) = x0;
x_est(:,1) = x0;

y_true(:,1) = C * x_true(:,1);  % 
y_meas(:,1) = y_true(:,1) + meas_noise_std * randn(ny,1);


% Reset baseline model state before simulation
fprintf('Resetting baseline model state...\n');
try
    baseline_inference.reset_state();
    fprintf('Baseline state reset successful.\n');
catch ME_reset
    fprintf('Warning: reset_state failed: %s\n', ME_reset.message);
    % If reset_state doesn't exist, that's okay - get_x will initialize
end

% Debug: Print first few inputs
fprintf('First 5 control inputs (scaled): ');
for t = 1:min(5, sim_length)
    u_cl(:,t) = controller{x_est(:,t)};
    fprintf('%.4f ', u_cl(:,t));
end
fprintf('\n');

for t = 1:sim_length
    if t == 1
        u_cl(:,t) = controller{x_est(:,t)};%toto zistit ci sem ide u_cl/u_est
    else
        u_cl(:,t) = controller{x_est(:,t)};
    end

    % === BASELINE INFERENCE INTEGRATION ===
    % Use baseline model for true system dynamics
    if t == 1
        % Initialize baseline model with current state (scaled)
        fprintf('Initializing baseline with y0 = %.4f (scaled), %.4f °C (descaled)\n', ...
            y_true(:,t), y_true(:,t) * x_std + x_mean);
        baseline_inference.get_x(y_true(:,t));
    end
    
    % Get measurement from baseline model (true system)
    % u_cl is already in scaled domain
    if t <= 3
        fprintf('t=%d: u_cl=%.4f (scaled), %.4f%% (descaled) -> ', ...
            t, u_cl(:,t), u_cl(:,t) * u_std + u_mean);
    end
    y_baseline = baseline_inference.y_plus(u_cl(:,t));
    y_baseline_array = double(y_baseline);
    y_true(:,t+1) = y_baseline_array(1); % Extract scalar value
    if t <= 3
        fprintf('y_baseline=%.4f (scaled), %.4f °C (descaled)\n', ...
            y_baseline_array(1), y_baseline_array(1) * x_std + x_mean);
    end
  
    %noised
    y_meas(:,t+1) = y_true(:,t+1) + meas_noise_std * randn(ny,1);


    %tu filter x_KF=...
    x_pred = A * x_est(:,t) + B * u_cl(:,t);         % x_{t+1|t}
    P_pred = A * P * A' + Q_kalman;                  % P_{t+1|t}
    
    %vykreslit x_KF, x_cl
     % --- Kalman gain and update using measurement y_meas(:,t+1) ---
    S = C * P_pred * C' + R_kalman;                  % innovation covariance (scalar)
    K = (P_pred * C') / S;                           % Kalman gain (nx x ny)
    x_est(:,t+1) = x_pred + K * (y_meas(:,t+1) - C * x_pred); %x est
    P = (eye(nx) - K * C) * P_pred;
    
    % (optional) keep P positive definite numerically
    P = (P + P')/2;
end

% Descale
y_true_desc = y_true * x_std + x_mean; % cl
u_cl_desc = u_cl * u_std + u_mean;
y_est_desc = x_est * x_std + x_mean;  %kf estimation (using Strejc scaling)
%% --- Plot closed-loop only ---
time = 0:sim_length;
figure;
subplot(3,1,1)
plot(time, y_true_desc, 'm-', 'LineWidth', 1.5); hold on
plot(time, y_est_desc, 'b--', 'LineWidth', 2.5);%observer
xlabel('Time step'); ylabel('Output y (°C)');
legend('True output (Baseline)','KF Estimate (Strejc)');
yline(x_mean)
title('Strejc MPC + Kalman Filter vs Baseline System');
grid on;grid minor;
ylim([40 70])

subplot(3,1,2)
stairs(time(1:end-1), u_cl_desc, 'k--', 'LineWidth', 2);
xlabel('Time step'); ylabel('Input u');
title('MPC Input');
grid on;grid minor;

subplot(3,1,3)
% plot measurement (noisy) and true
plot(time, y_true_desc, 'm--', 'LineWidth', 1.5); hold on
plot(time, y_est_desc, 'b-', 'LineWidth', 1.5);
plot(time, (y_meas * x_std + x_mean), 'gx'); % noisy measurements (descaled)
xlabel('Time step'); ylabel('Output y (°C)');
legend('Plant true', 'KF estimate', 'Noisy measurements');
title('Measurements vs KF');
grid on;grid minor;
ylim([40 70])


%% --- Save and RMSEC ---
rmse_strejc_to_zero = sqrt(mean((y_true_desc(:)).^2)); % RMSE to zero °C
fprintf('RMSE (Strejc to 0°C) = %.4f °C\n', rmse_strejc_to_zero);

save('results_strejc_to_zero.mat', 'y_true_desc', 'y_est_desc', 'u_cl_desc','u_cl','y_true');
save('baseline_reference_strejc.mat', 'y_true_desc');  % Save Strejc baseline for comparison
