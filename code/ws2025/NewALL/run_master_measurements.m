%% RUN_MASTER_MEASUREMENTS
% Spustí sekvenciu meraní pre viac počiatočných teplôt.
% Pre každý cieľ prehriatia najprv spustí Koopman regulátor a následne Strejc.

initial_temperatures = [45, 50, 55, 58, 60, 62, 66, 68];
default_hold_loops = 5;

script_dir = fileparts(mfilename('fullpath'));
koop_script = fullfile(script_dir, 'control_koop_wC_ys_master.m');
strejc_script = fullfile(script_dir, 'control_strejc_wC_ys_master.m');
data_dir = fullfile(script_dir, 'steps');

if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

skip_pct23_off = true; % nechávame zariadenie zapnuté až do konca master skriptu

for idx = 1:numel(initial_temperatures)
    temp_target = initial_temperatures(idx);
    measurement_label = sprintf('meas%02d_T%02.0f', idx, temp_target);
    fprintf('\n=== Začínam meranie %s (setpoint %.2f °C) ===\n', measurement_label, temp_target);

    % --- Koopman sekcia ---
    setpoint_preheat = temp_target;
    hold_loops = default_hold_loops;
    filename = fullfile(data_dir, sprintf('koop_%s.mat', measurement_label));
    run(koop_script);

    % --- Strejc sekcia ---
    setpoint_preheat = temp_target;
    hold_loops = default_hold_loops;
    filename = fullfile(data_dir, sprintf('strejc_%s.mat', measurement_label));
    run(strejc_script);
end

skip_pct23_off = false;
if exist('pct23', 'var') && ~isempty(pct23)
    try
        pct23.off();
        pct23.setTag('FSV', 1);
    catch pct_err
        warning('Nepodarilo sa vypnúť pct23: %s', pct_err.message);
    end
end

clear setpoint_preheat hold_loops filename temp_target measurement_label skip_pct23_off;

