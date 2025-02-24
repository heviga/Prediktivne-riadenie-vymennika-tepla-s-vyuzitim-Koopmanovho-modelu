function result = strejc_controll(y)
    u_opt = py.koopman_mpc.get_strejc_u(y);
    result = double(u_opt);
end
