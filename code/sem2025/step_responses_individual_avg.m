clc; close all;
data1 = load('ident_data_1.mat');
data2 = load('ident_data_2_9_steps.mat');

% Teploty
T4_1 = data1.Temperatures{4}.Values.Data;
T4_2 = data2.Temperatures{4}.Values.Data;

% Časy
t1 = data1.tout;
t2 = data2.tout;

% Input
u1 = data1.uout(:,2);
u2 = data2.uout(:,2);

% Kedy step change
step_indices_1 = find(abs(diff(u1)) > 1);
step_indices_1 = step_indices_1(2:end);
step_indices_2 = find(abs(diff(u2)) >= 1);
step_indices_2 = step_indices_2(2:end-2);

% Merge
t = [t1; t2];
x = [T4_1; T4_2];
u = [u1; u2];

% Shift step casy druheho merania
step_indices_2 = step_indices_2 + length(t1);
step_indices = [step_indices_1; step_indices_2];
num_steps = length(step_indices);

% Uloženie jednotlivých krokov do vektorov
t_steps = cell(num_steps, 1);
x_steps = cell(num_steps, 1);
u_steps = cell(num_steps, 1);

for i = 1:num_steps
    start_idx = step_indices(i);
    
    if i < num_steps
        end_idx = step_indices(i+1) - 1;
    else
        end_idx = start_idx + 250;
    end
    
    % Obmedzenie 12. kroku na 250s
    if i == 12
        end_idx = start_idx + 250;
    end
    
    t_steps{i} = t(start_idx:end_idx) - t(start_idx);
    x_steps{i} = x(start_idx:end_idx);
    u_steps{i} = u(start_idx:end_idx);
end

% Vykreslenie jednotlivých krokov v samostatných figurách
for i = 1:num_steps
    figure;
    subplot(2,1,1);
    plot(t_steps{i}, x_steps{i}, 'b', 'LineWidth', 1.5);
    ylabel('T4 (°C)');
    xlabel('Čas (s)');
    title(['Krok ', num2str(i), ' - Teplota']);
    grid on;
    
    subplot(2,1,2);
    plot(t_steps{i}, u_steps{i}, 'r', 'LineWidth', 1.5);
    ylabel('u (%)');
    xlabel('Čas (s)');
    title(['Krok ', num2str(i), ' - Input']);
    grid on;
end


% Vykreslenie všetkých normalizovaných step response do jednej figúry
figure; hold on;
colors = lines(num_steps); % Generovanie rôznych farieb

for i = 1:num_steps
    % Normalizácia
    step_size = u_steps{i}(2) - u_steps{i}(1);
    x_norm = (x_steps{i} - x_steps{i}(1)) / abs(step_size);
    
    % Ak je step change negatívny, invertujeme odpoveď
    if step_size < 0
        x_norm = -x_norm;
    end
    
    plot(t_steps{i}, x_norm, 'Color', colors(i,:), 'LineWidth', 1.5);
end

title('Normalizované Step Responses');
xlabel('Čas (s)');
ylabel('Normalizovaná teplota');
grid on;
hold off;



% Výpočet priemernej step response
max_length = max(cellfun(@length, x_steps));
x_matrix = NaN(num_steps, max_length);

for i = 1:num_steps
    step_size = u_steps{i}(2) - u_steps{i}(1);
    x_norm = (x_steps{i} - x_steps{i}(1)) / abs(step_size);
    if step_size < 0
        x_norm = -x_norm;
    end
    x_matrix(i, 1:length(x_norm)) = x_norm;
end

avg_response = nanmean(x_matrix, 1);

figure;
plot(1:max_length, avg_response, 'k', 'LineWidth', 2);
title('Priemerná Normalizovaná Step Response');
xlabel('Časový Index');
ylabel('Normalizovaná Teplota');
xlim([0,250])
grid on;



