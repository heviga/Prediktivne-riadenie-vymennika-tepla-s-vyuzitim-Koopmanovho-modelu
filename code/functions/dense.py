import numpy as np

# Creating of the matrices for dense formulation of the MPC problem with refence tracking and delta u formulation
# The matrices are created according to the Martin Klaučo minimal thesis p. 23-25

def create_weighting(Q, N):
    n = Q.shape[0]
    W = np.zeros((n * N, n  * N))
    for i in range(N):
        W[i * n:(i + 1) * n, i * n:(i + 1) * n] = Q
    return W

def create_Psi_y(A, C, N):
    ny = C.shape[0]
    nx = C.shape[1]
    Psi_y = np.zeros((ny * N, nx))
    for i in range(N):
        Psi_y[i*ny:i*ny+ny, :] = C @ np.linalg.matrix_power(A, i)
    return Psi_y

def create_Gamma_y(A, B, C, D, N):
    ny = D.shape[0]
    nu = D.shape[1]
    Gamma_y = np.zeros((ny*N, nu*N))
    Gamma_y[:ny, :nu] = D
    for i in range(1, N):
        Gamma_y[i*ny:i*ny+ny, i*nu:i*nu+nu] = D
        for j in range(1, i+1):
            Gamma_y[i*ny:i*ny+ny, (i-j)*nu:(i-j)*nu+nu] = C @ np.linalg.matrix_power(A, j-1) @ B
    return Gamma_y

def create_Lambda(nu, N):
    Lambda = np.zeros((nu*N, nu*N))
    for i in range(0, N):
        Lambda[i*nu:i*nu+nu, i*nu:i*nu+nu] = np.eye(nu)
        if i > 0:
            Lambda[i*nu:i*nu+nu, i*nu-nu:i*nu] = -np.eye(nu)
    return Lambda

def create_lambda(nu, N):
    lambda_ = np.zeros((nu*N, nu))
    lambda_[:nu, :] = -np.eye(nu)
    return lambda_

def create_N_vector(N, vector):
    vector = np.tile(vector, (N, 1))
    return vector

def create_Psi_x(A,N):
    N=N+1
    nx = A.shape[0]
    Psi_x = np.zeros((N*nx, nx))
    for i in range(0, N):
        Psi_x[i*nx:i*nx+nx,:] = np.linalg.matrix_power(A, i)
    return Psi_x

def create_Psi_u(A,B,N):
    N=N+1
    nx = A.shape[0]
    nu = B.shape[1]
    Psi_u = np.zeros((N*nx, (N-1)*nu))
    for i in range(1, N):
        for j in range(1, i+1):
            Psi_u[i*nx:(i+1)*nx,(j-1)*nu:j*nu] = np.linalg.matrix_power(A, i-j) @ B
    return Psi_u

# State regulation

def quad_form_sr(A, B, Qx, Qu, N):
    Psi_u = create_Psi_u(A,B,N)
    Wx = create_weighting(Qx, N+1)
    Wu = create_weighting(Qu, N)
    return Psi_u.T@Wx@Psi_u + Wu

def lin_form_sr(A, B, Qx, N, x0):
    Psi_u = create_Psi_u(A,B,N)
    Psi_x = create_Psi_x(A,N)
    Wx = create_weighting(Qx, N+1)
    return 2*x0.T@Psi_x.T@Wx@Psi_u

def constraint_matrix_sr(A,B,N):
    Psi_u = create_Psi_u(A,B,N)
    I = np.eye(N)
    return np.vstack((Psi_u, -Psi_u, I, -I))

def upper_bound_sr(A,N, x0, xmax, xmin, umax, umin):
    Psi_x = create_Psi_x(A,N)
    Xmax = create_N_vector(N+1, xmax)
    Xmin = create_N_vector(N+1, xmin)
    Umax = create_N_vector(N, umax)
    Umin = create_N_vector(N, umin)
    return np.vstack((Xmax-Psi_x@x0, -Xmin+Psi_x@x0, Umax, -Umin))

def constraint_matrix_sr_nox(B, N):
    I = np.eye(N*B.shape[1])
    return np.vstack((I, -I))

def upper_bound_sr_nox(N, umax, umin):
    Umax = create_N_vector(N, umax)
    Umin = create_N_vector(N, umin)
    return np.vstack((Umax, -Umin))

# State control to reference

def quad_form_cr(A, B, Qx, Qu, N):
    Psi_u = create_Psi_u(A,B,N)    
    Lambda = create_Lambda(B.shape[1], N)
    Wx = create_weighting(Qx, N+1)
    Wu = create_weighting(Qu, N)
    return Psi_u.T@Wx@Psi_u + Lambda.T @ Wu @ Lambda

def lin_form_cr(A, B, Qx, Qu, N, x0, x_ref, u_prev):
    Psi_u = create_Psi_u(A,B,N)
    Psi_x = create_Psi_x(A,N)
    Lambda = create_Lambda(B.shape[1], N)
    lambda_ = create_lambda(B.shape[1], N)
    Wx = create_weighting(Qx, N+1)
    Wu = create_weighting(Qu, N)
    X_ref = create_N_vector(N+1, x_ref)
    return 2*x0.T@Psi_x.T@Wx@Psi_u + 2*u_prev.T@lambda_.T@Wu@Lambda - 2*X_ref.T@Wx@Psi_u



# Output regulation

def quad_form_or(A, B, C, D, Qy, Qu, N):
    Gamma_y = create_Gamma_y(A, B, C, D, N)
    Wy = create_weighting(Qy, N)
    Wu = create_weighting(Qu, N)
    return Gamma_y.T@Wy@Gamma_y + Wu

def lin_form_or(A, B, C, D, Qy, N, x0):
    Gamma_y = create_Gamma_y(A, B, C, D, N)
    Psi_y = create_Psi_y(A, C ,N)
    Wy = create_weighting(Qy, N)
    return 2*x0.T@Psi_y.T@Wy@Gamma_y

def constraint_matrix_or(A,B,C,D,N):
    Gamma_y = create_Gamma_y(A, B, C, D, N)
    I = np.eye(N)
    return np.vstack((Gamma_y, -Gamma_y, I, -I))

def upper_bound_or(A, C, N, x0, ymax, ymin, umax, umin):
    Psi_y = create_Psi_y(A, C ,N)
    Ymax = create_N_vector(N, ymax)
    Ymin = create_N_vector(N, ymin)
    Umax = create_N_vector(N, umax)
    Umin = create_N_vector(N, umin)
    return np.vstack((Ymax-Psi_y@x0, -Ymin+Psi_y@x0, Umax, -Umin))

def constraint_matrix_or_nc(B,N):
    I = np.eye(N*B.shape[1])
    return np.vstack((I, -I))

def upper_bound_or_nc(umax, umin, N):
    Umax = create_N_vector(N, umax)
    Umin = create_N_vector(N, umin)
    return np.vstack(( Umax, -Umin))


# Output control to reference

def quad_form(A, B, C, D, Qy, Qu, N):
    Gamma_y = create_Gamma_y(A, B, C, D, N)
    Lambda = create_Lambda(B.shape[1], N)
    Wy = create_weighting(Qy, N)
    Wu = create_weighting(Qu, N)
    return Gamma_y.T @ Wy @ Gamma_y + Lambda.T @ Wu @ Lambda

def lin_form(A, B, C, D, Qy, Qu, N, x0, y_ref, u_prev):
    Gamma_y = create_Gamma_y(A, B, C, D, N)
    Psi_y = create_Psi_y(A, C, N)
    Lambda = create_Lambda(B.shape[1], N)
    lambda_ = create_lambda(B.shape[1], N)
    Wy = create_weighting(Qy, N)
    Wu = create_weighting(Qu, N)
    Y_ref = create_N_vector(N, y_ref)
    return 2 * x0.T @ Psi_y.T @ Wy @ Gamma_y + 2 * u_prev.T @ lambda_.T @ Wu @ Lambda - 2 * Y_ref.T @ Wy @ Gamma_y

def constraint_matrix(A, B, C, D, N):
    Gamma_y = create_Gamma_y(A, B, C, D, N)
    Lambda = create_Lambda(B.shape[1], N)
    INnu = np.eye(B.shape[1]*N)
    return np.vstack((Gamma_y, -Gamma_y,  Lambda, -Lambda, INnu, -INnu))

def upper_bound(A, B, C, x0, u_prev, ymax, ymin, d_umax, d_umin, umax, umin, N):
    Psi_y = create_Psi_y(A, C, N)
    lambda_ = create_lambda(B.shape[1], N)
    Ymax = create_N_vector(N, ymax)
    Ymin = create_N_vector(N, ymin)
    d_Umax = create_N_vector(N, d_umax)
    d_Umin = create_N_vector(N, d_umin)
    Umax = create_N_vector(N, umax)
    Umin = create_N_vector(N, umin)
    return np.vstack((Ymax - Psi_y @ x0, -Ymin + Psi_y @ x0, + d_Umax - lambda_ @ u_prev , - d_Umin + lambda_ @ u_prev , Umax, -Umin))

def constraint_matrix_no(B, N):
    INnu = np.eye(N*B.shape[1])
    return np.vstack((INnu, -INnu))

def upper_bound_no(umax, umin, N):
    Umax = create_N_vector(N, umax)
    Umin = create_N_vector(N, umin)
    return np.vstack((Umax, -Umin))

# No state constraints

def constraint_matrix_nsc(B,  N):
    Lambda = create_Lambda(B.shape[1], N)
    INnu = np.eye(B.shape[1]*N)
    return np.vstack((  Lambda, -Lambda, INnu, -INnu))

def upper_bound_nsc(B, u_prev, d_umax, d_umin, umax, umin, N):
    lambda_ = create_lambda(B.shape[1], N)
    d_Umax = create_N_vector(N, d_umax)
    d_Umin = create_N_vector(N, d_umin)
    Umax = create_N_vector(N, umax)
    Umin = create_N_vector(N, umin)
    return np.vstack((d_Umax - lambda_ @ u_prev , - d_Umin + lambda_ @ u_prev , Umax, -Umin))
