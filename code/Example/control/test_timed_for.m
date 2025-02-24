N = 800; % Number of iterations

for i = 1:N
    % Start the timer
    tic;

    % Code to execute within the loop
    disp(['Iteration: ', num2str(i), ' at ', datestr(now)]);
    
    koopman_controll_noC(50, "xs", 50)

    % Wait until 1 second has passed since the start of the iteration
    elapsedTime = toc;
    pause(max(0, 1 - elapsedTime));
end