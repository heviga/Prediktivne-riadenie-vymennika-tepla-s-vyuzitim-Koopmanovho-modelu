function result = koopman_controll(y)
    u_opt = py.koopman_mpc.get_koopman_u(y);
    result = double(u_opt);
end
