%% Estimate settling band from identification data
% settling_band = 2 * sigma_noise
clc; clear; close all;

project_root = 'C:\Users\ivadu\Desktop\9.semestrik\vymennik\Prediktivne-riadenie-vymennika-tepla-s-vyuzitim-Koopmanovho-modelu\code\LS2026';
addpath(genpath(project_root));

%% ===== LOAD IDENTIFICATION DATA =====
data = load(fullfile(project_root, 'identification', 'identifikacia2.mat'));

% Adjust these lines if variable names differ in your file
y = data.Temperatures{4}.Values.Data;   % T4
u = data.uout(:,2);                     % Pump2
t = data.tout;

% Remove initial bad part if you use the same preprocessing elsewhere
u = u(251:end);
y = y(251:end);
t = t(251:end);

% Reset time
t = t - t(1) + 1;

%% ===== BASIC PLOT =====
figure('Color','w','Position',[100 100 900 500]);
subplot(2,1,1);
plot(t, y, 'LineWidth', 1.2);
grid on;
ylabel('T4 (^oC)');
title('Identification data');

subplot(2,1,2);
plot(t, u, 'LineWidth', 1.2);
grid on;
ylabel('Pump2 (%)');
xlabel('Time (s)');

%% ===== FIND STEP CHANGES IN INPUT =====
du = [0; diff(u)];
step_threshold = 5;     % change if needed
step_idx = find(abs(du) > step_threshold);

fprintf('Detected %d input changes.\n', length(step_idx));
disp('Step indices:');
disp(step_idx.');

figure('Color','w','Position',[100 100 900 300]);
plot(t, u, 'LineWidth', 1.2); hold on;
xline(t(step_idx), 'r--');
grid on;
ylabel('Pump2 (%)');
xlabel('Time (s)');
title('Detected input changes');

%% ===== METHOD 1: SIGMA FROM WHOLE DETRENDED SIGNAL =====
% Remove a smooth trend and estimate residual std
win = 81;  % odd number
y_smooth = movmean(y, win);
resid_all = y - y_smooth;
sigma_all = std(resid_all);

%% ===== METHOD 2: SIGMA FROM LOW-SLOPE REGIONS =====
dy = [0; diff(y)];
slope_threshold = 0.02;  % degC/sample
idx_flat = abs(dy) < slope_threshold;
sigma_flat = std(y(idx_flat) - mean(y(idx_flat)));

%% ===== METHOD 3: SIGMA FROM STEADY-STATE SEGMENTS BETWEEN STEPS =====
% This is usually the most useful for settling band selection
guard_after_step = 40;   % ignore transient immediately after step
tail_len = 30;           % take last 30 samples before next step

steady_idx = false(size(y));

if isempty(step_idx)
    warning('No steps detected. Falling back to low-slope estimate only.');
else
    % segment before first detected step
    if step_idx(1) > tail_len
        idx0 = max(1, step_idx(1)-tail_len):step_idx(1)-1;
        steady_idx(idx0) = true;
    end

    % segments between steps
    for i = 1:length(step_idx)-1
        seg_start = step_idx(i) + guard_after_step;
        seg_end   = step_idx(i+1) - 1;

        if seg_end - seg_start + 1 >= tail_len
            idx_seg = seg_end-tail_len+1 : seg_end;
            steady_idx(idx_seg) = true;
        end
    end

    % segment after last detected step
    seg_start = step_idx(end) + guard_after_step;
    seg_end   = length(y);

    if seg_end - seg_start + 1 >= tail_len
        idx_seg = seg_end-tail_len+1 : seg_end;
        steady_idx(idx_seg) = true;
    end
end

y_steady = y(steady_idx);
sigma_steady = std(y_steady);

%% ===== OPTIONAL: LOCAL SEGMENT-WISE SIGMAS =====
segment_sigmas = [];
if any(steady_idx)
    % split contiguous steady-state regions
    dsteady = diff([false; steady_idx; false]);
    seg_starts = find(dsteady == 1);
    seg_ends   = find(dsteady == -1) - 1;

    for i = 1:length(seg_starts)
        yi = y(seg_starts(i):seg_ends(i));
        if numel(yi) >= 5
            segment_sigmas(end+1,1) = std(yi); %#ok<SAGROW>
        end
    end
end

sigma_segments_mean = mean(segment_sigmas, 'omitnan');
sigma_segments_max  = max(segment_sigmas, [], 'omitnan');

%% ===== CHOOSE FINAL SIGMA =====
% Recommended choice:
sigma_noise = sigma_steady;

settling_band = 2 * sigma_noise;

%% ===== PRINT RESULTS =====
fprintf('\n===== SIGMA ESTIMATES =====\n');
fprintf('Sigma from detrended whole signal      : %.4f °C\n', sigma_all);
fprintf('Sigma from low-slope regions           : %.4f °C\n', sigma_flat);
fprintf('Sigma from steady-state segments       : %.4f °C\n', sigma_steady);
fprintf('Mean sigma across steady-state segments: %.4f °C\n', sigma_segments_mean);
fprintf('Max sigma across steady-state segments : %.4f °C\n', sigma_segments_max);

fprintf('\n===== RECOMMENDED SETTLING BAND =====\n');
fprintf('sigma_noise   = %.4f °C\n', sigma_noise);
fprintf('settling_band = 2*sigma = %.4f °C\n', settling_band);

%% ===== VISUALIZE CHOSEN STEADY-STATE SAMPLES =====
figure('Color','w','Position',[100 100 900 500]);

subplot(2,1,1);
plot(t, y, 'b', 'LineWidth', 1.0); hold on;
plot(t(steady_idx), y(steady_idx), 'ro', 'MarkerSize', 4, 'LineWidth', 1.0);
grid on;
ylabel('T4 (^oC)');
title(sprintf('Steady-state samples used for sigma estimate, sigma = %.4f ^oC', sigma_steady));
legend('T4','Selected steady-state samples','Location','best');

subplot(2,1,2);
plot(t, u, 'k', 'LineWidth', 1.0); hold on;
xline(t(step_idx), 'r--');
grid on;
ylabel('Pump2 (%)');
xlabel('Time (s)');
title('Input with detected step changes');

%% ===== SAVE RESULT =====
save(fullfile(project_root, 'results', 'settling_band_from_identification.mat'), ...
    'sigma_all', 'sigma_flat', 'sigma_steady', ...
    'sigma_segments_mean', 'sigma_segments_max', ...
    'sigma_noise', 'settling_band', ...
    'step_idx', 'steady_idx');

fprintf('\nSaved result to:\n%s\n', ...
    fullfile(project_root, 'results', 'settling_band_from_identification.mat'));