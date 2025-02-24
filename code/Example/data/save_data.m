%save("/Users/patrik/Documents/Research/2024-2025/koopman_mpc_delay_turbine/code/HE_real/Data/T50-d05-5m.mat","Temperatures")
%save("/Users/patrik/Documents/Research/2024-2025/koopman_mpc_delay_turbine/code/HE_real/Data/u50-d05-5m.mat","uout")

T4 = Temperatures{4}.Values.Data
u = 
save("/Users/patrik/Documents/Research/2024-2025/koopman_mpc_delay_turbine/code/HE_real/Data/T4_sim.mat","T4")
save("/Users/patrik/Documents/Research/2024-2025/koopman_mpc_delay_turbine/code/HE_real/Data/u_sim.mat","u")

