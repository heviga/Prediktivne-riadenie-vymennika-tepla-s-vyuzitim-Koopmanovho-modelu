function result = koopman_controll_noC(y, xref, u_prev)
    u_opt = py.koopman_mpc.get_koopman_u_noC(y, xref, u_prev);
    result = double(u_opt);
end

