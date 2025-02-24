function result = strejc_controll_ref(y, ref, u_prev)
    u_opt = py.koopman_mpc.get_strejc_u(y, ref, u_prev);
    result = double(u_opt);
end
