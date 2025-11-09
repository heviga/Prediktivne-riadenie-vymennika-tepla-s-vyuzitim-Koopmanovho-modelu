function result = koopman_controll(y,uprev)
    u_opt = py.koopman_mpc.get_koopman_u_wC(y,uprev,0);
    result = double(u_opt);
end

