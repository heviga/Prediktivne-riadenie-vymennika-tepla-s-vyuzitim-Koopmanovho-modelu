% Test script to verify baseline model responds to different inputs
clear all; close all; clc;

% Configure Python environment
python_exe = 'C:\Users\ivadu\AppData\Local\Programs\Python\Python39\python.exe';
python_path = 'C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\ws2025\NewALL';

% Initialize Python
pyenv('Version', python_exe);
if count(py.sys.path, python_path) == 0
    py.sys.path().append(python_path);
end

% Import baseline inference
baseline_inference = py.importlib.import_module('baseline_inference');
baseline_inference.init();

% Load scaling parameters
load('train_data.mat');
load('test_data.mat');
Yall = [Ytrain(:); Ytest(:)];
Uall = [Utrain(:); Utest(:)];
x_mean = mean(Yall);
x_std = std(Yall);
u_mean = mean(Uall);
u_std = std(Uall);

% Test 1: Same initial condition, different inputs
fprintf('=== Test 1: Same initial condition, different inputs ===\n');
y0_scaled = (50 - x_mean) / x_std;

% Reset and test with input u1
baseline_inference.reset_state();
baseline_inference.get_x(y0_scaled);
u1_scaled = (60 - u_mean) / u_std;  % Input 60%
y1 = baseline_inference.y_plus(u1_scaled);
y1_array = double(y1);  % Convert to MATLAB array first
y1_desc = y1_array(1) * x_std + x_mean;
fprintf('Input u1 = 60%%, Output y1 = %.4f °C\n', y1_desc);

% Reset and test with input u2
baseline_inference.reset_state();
baseline_inference.get_x(y0_scaled);
u2_scaled = (80 - u_mean) / u_std;  % Input 80%
y2 = baseline_inference.y_plus(u2_scaled);
y2_array = double(y2);  % Convert to MATLAB array first
y2_desc = y2_array(1) * x_std + x_mean;
fprintf('Input u2 = 80%%, Output y2 = %.4f °C\n', y2_desc);

if abs(y1_desc - y2_desc) < 0.01
    warning('PROBLEM: Different inputs produce same output!');
else
    fprintf('✓ Baseline model responds correctly to different inputs\n');
end

% Test 2: Simulate a few steps with different input sequences
fprintf('\n=== Test 2: Different input sequences ===\n');
baseline_inference.reset_state();
baseline_inference.get_x(y0_scaled);

% Sequence 1: Low inputs
u_seq1 = [(50 - u_mean)/u_std, (55 - u_mean)/u_std, (60 - u_mean)/u_std];
y_seq1 = zeros(1, 4);
y_seq1(1) = y0_scaled;
for i = 1:3
    y_temp = baseline_inference.y_plus(u_seq1(i));
    y_temp_array = double(y_temp);  % Convert to MATLAB array first
    y_seq1(i+1) = y_temp_array(1);
end
y_seq1_desc = y_seq1 * x_std + x_mean;
fprintf('Sequence 1 (50, 55, 60%%): Final output = %.4f °C\n', y_seq1_desc(end));

% Reset and test sequence 2
baseline_inference.reset_state();
baseline_inference.get_x(y0_scaled);

% Sequence 2: High inputs
u_seq2 = [(80 - u_mean)/u_std, (85 - u_mean)/u_std, (90 - u_mean)/u_std];
y_seq2 = zeros(1, 4);
y_seq2(1) = y0_scaled;
for i = 1:3
    y_temp = baseline_inference.y_plus(u_seq2(i));
    y_temp_array = double(y_temp);  % Convert to MATLAB array first
    y_seq2(i+1) = y_temp_array(1);
end
y_seq2_desc = y_seq2 * x_std + x_mean;
fprintf('Sequence 2 (80, 85, 90%%): Final output = %.4f °C\n', y_seq2_desc(end));

if abs(y_seq1_desc(end) - y_seq2_desc(end)) < 0.01
    warning('PROBLEM: Different input sequences produce same final output!');
else
    fprintf('✓ Baseline model responds correctly to different input sequences\n');
end

fprintf('\n=== Test Complete ===\n');
fprintf('If both tests passed, baseline model is working correctly.\n');
fprintf('You need to RE-RUN cl_koopman and cl_strejc to generate new results.\n');

