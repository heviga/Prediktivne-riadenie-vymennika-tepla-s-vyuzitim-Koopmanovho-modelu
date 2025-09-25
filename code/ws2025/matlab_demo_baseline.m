terminate(pyenv);
pyenv('Version', 'C:\Users\ivadu\AppData\Local\Programs\Python\Python39\python.exe');

%%pyenv('Version','/Users/patrik/miniconda3/envs/neuromancer/bin/python'); % here chenge the path to your library

py.baseline_inference.init()

sim_length = 500;

y_sim = zeros(1, sim_length+1);
u_sim = ones(1, sim_length);

y0 = 0;
py.baseline_inference.get_x(y0)
y_sim(1) = y0;

for i = 1:sim_length
    y_sim(i+1) = py.baseline_inference.y_plus(u_sim(i));
end

plot(y_sim)