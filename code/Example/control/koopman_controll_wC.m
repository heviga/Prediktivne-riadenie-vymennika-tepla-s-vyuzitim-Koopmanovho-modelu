function result = koopman_controll_wC(y, ref, u_prev)
    u_opt = py.koopman_mpc.get_koopman_u_wC(y, ref, u_prev);
    result = double(u_opt);
end

