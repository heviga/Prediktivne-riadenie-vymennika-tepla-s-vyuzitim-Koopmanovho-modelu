import sys
sys.path.append('../../functions')
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle 
import scipy
import time as tim

from neuromancer.psl import plot
from neuromancer import psl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from neuromancer.system import Node, System
from neuromancer.slim import slim
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer. modules import blocks
import joblib

from sklearn.preprocessing import StandardScaler
torch.manual_seed(0)

import numpy as np
from scipy.linalg import expm
import dense
import cvxpy as cp
import matplotlib.pyplot as plt

from scipy.signal import cont2discrete

class Koopman_control(nn.Module):
    """
    Baseline class for Koopman control model
    Implements discrete-time dynamical system:
        x_k+1 = K x_k + u_k
    with variables:
        x_k - latent states
        u_k - latent control inputs
    """

    def __init__(self, K):
        super().__init__()
        self.K = K

    def forward(self, x, u):
        """
        :param x: (torch.Tensor, shape=[batchsize, nx])
        :param u: (torch.Tensor, shape=[batchsize, nx])
        :return: (torch.Tensor, shape=[batchsize, nx])
        """
        x = self.K(x) + u
        return x

def load_problems():
    nx_koopman = 10
    layers = [60, 60, 60]
    ny = 1
    nsteps = 80   
    nu = 1 
    global problem, problem_noC
    
    # instantiate output encoder neural net f_y
    f_y = blocks.MLP(
        ny,
        nx_koopman,
        bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ReLU,
        hsizes=layers,
    )
    # initial condition encoder
    encode_Y0 = Node(f_y, ['Y0'], ['x'], name='encoder_Y0')
    # observed trajectory encoder
    encode_Y = Node(f_y, ['Y'], ['x_latent'], name='encoder_Y')
    
        
    f_u = torch.nn.Linear(nu, nx_koopman, bias=False)
    # initial condition encoder
    encode_U = Node(f_u, ['U'], ['u_latent'], name='encoder_U')
    
    f_y_inv = torch.nn.Linear(nx_koopman, ny, bias=False)
    # predicted trajectory decoder
    decode_y = Node(f_y_inv, ['x'], ['yhat'], name='decoder_y')
    
    #noC
    # noC

    # instantiate output encoder neural net f_y
    f_y_noC = blocks.MLP(
        ny,
        nx_koopman,
        bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ReLU,
        hsizes=layers,
    )
    # initial condition encoder
    encode_Y0_noC = Node(f_y_noC, ['Y0'], ['x'], name='encoder_Y0')
    # observed trajectory encoder
    encode_Y_noC = Node(f_y_noC, ['Y'], ['x_latent'], name='encoder_Y')
    # instantiate input encoder net f_u

    f_u_noC = torch.nn.Linear(nu, nx_koopman, bias=False)
    # initial condition encoder
    encode_U_noC = Node(f_u_noC, ['U'], ['u_latent'], name='encoder_U')

    # instantiate state decoder neural net f_y_inv
    f_y_inv_noC = blocks.MLP(nx_koopman, ny, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ELU,
                    hsizes=layers)
    #f_y_inv = torch.nn.Linear(nx_koopman, ny, bias=False)
    # predicted trajectory decoder
    decode_y_noC = Node(f_y_inv_noC, ['x'], ['yhat'], name='decoder_y')
    
    K = torch.nn.Linear(nx_koopman, nx_koopman, bias=False)
    K_noC = torch.nn.Linear(nx_koopman, nx_koopman, bias=False)
    
    # symbolic Koopman model with control inputs
    Koopman = Node(Koopman_control(K), ['x', 'u_latent'], ['x'], name='K')
    Koopman_noC = Node(Koopman_control(K_noC), ['x', 'u_latent'], ['x'], name='K')

    # latent Koopmann rollout
    dynamics_model = System([Koopman], name='Koopman', nsteps=nsteps)
    dynamics_model_noC = System([Koopman_noC], name='Koopman', nsteps=nsteps)
    
    nodes = [encode_Y0, encode_Y, encode_U, dynamics_model, decode_y]
    nodes_noC = [encode_Y0_noC, encode_Y_noC, encode_U_noC, dynamics_model_noC, decode_y_noC]
    
    # variables
    Y = variable("Y")  # observed
    yhat = variable('yhat')  # predicted output
    x_latent = variable('x_latent')  # encoded output trajectory in the latent space
    u_latent = variable('u_latent')  # encoded input trajectory in the latent space
    x = variable('x')  # Koopman latent space trajectory

    xu_latent = x_latent + u_latent  # latent state trajectory

    # output trajectory tracking loss
    y_loss = 10. * (yhat[:, 1:-1, :] == Y[:, 1:, :]) ^ 2
    y_loss.name = "y_loss"

    # one-step tracking loss
    onestep_loss = 1.*(yhat[:, 1, :] == Y[:, 1, :])^2
    onestep_loss.name = "onestep_loss"

    # latent trajectory tracking loss
    x_loss = 1. * (x[:, 1:-1, :] == xu_latent[:, 1:, :]) ^ 2
    x_loss.name = "x_loss"

    objectives = [y_loss, x_loss, onestep_loss]
    loss = PenaltyLoss(objectives, constraints=[])
    problem = Problem(nodes, loss)
    problem_noC = Problem(nodes_noC, loss)

    problem.load_state_dict(torch.load('../data/modelC.pth'),strict=False)
    problem_noC.load_state_dict(torch.load('../data/model_noC.pth'),strict=False)

def generate_u(y):
    return 50

def get_x(y):
    y = np.array([[y]])
    x = problem.nodes[0]({"Y0": torch.from_numpy(y).float()})
    return x["x"][0].detach().numpy().reshape(1,-1).T

def get_x_noC(y):
    x = problem_noC.nodes[0]({"Y0": torch.from_numpy(y).float()})
    return x["x"].detach().numpy().reshape(1,-1).T

def get_koopman_u_wC(y, ref, u_prev):
    if ref == "ys":
        y_ref = ref_ys
    elif ref == "ref1":
        y_ref = ref_y_1
    elif ref == "ref2":
        y_ref = ref_y_2
    else:
        y_ref = np.array([[ref]])
        
    y_ref = scaler.transform(np.array(y_ref))
    u_prev = scalerU.transform(np.array([[u_prev]]))
    y_scaled = scaler.transform(np.array([[y]]))
    x0 = get_x(y_scaled)
    Q = dense.quad_form(A, B, C, D, Qy, Qu, N)                      
    c = dense.lin_form(A, B, C, D, Qy, Qu, N, x0, y_ref, u_prev)  
      
    F = dense.constraint_matrix_no(B, N)                   
    g = dense.upper_bound_no(umax, umin, N)
    
    U = cp.Variable((N, 1))
    
    objective = cp.Minimize(cp.quad_form(U, Q) + c @ U)
    constraints = [F @ U <= g]
    
    problem_mpc = cp.Problem(objective, constraints)
    problem_mpc.solve()
    
    return scalerU.inverse_transform(U.value[0, :].reshape(-1, 1))
    
def get_koopman_u_noC(y, ref, u_prev):
    if ref == "xs":
        x_ref = ref_xs
    elif ref == "ref1":
        x_ref = ref_x_1
    elif ref == "ref2":
        x_ref = ref_x_2
    else:
        print("There is an error")
        
    u_prev = scalerU.transform(np.array([[u_prev]]))
    y_scaled = scaler.transform(np.array([[y]]))
    x0 = get_x_noC(y_scaled)
    x_ref = np.array([x_ref]).T
    
    Q = dense.quad_form_cr(A_noC, B_noC, Qx, Qu, N)
    c = dense.lin_form_cr(A_noC, B_noC, Qx, Qu, N, x0, x_ref, u_prev)
    F = dense.constraint_matrix_sr_nox(N)
    g = dense.upper_bound_sr_nox(N,umax, umin)
    
    U = cp.Variable((N, 1))
    
    objective = cp.Minimize(cp.quad_form(U, Q) + c @ U)
    constraints = [F @ U <= g]
    
    problem_mpc = cp.Problem(objective, constraints)
    problem_mpc.solve()
    
    return scalerU.inverse_transform(U.value[0, :].reshape(-1, 1))
        

def get_strejc_u(y, ref, u_prev):
    if ref == "ys":
        y_ref = ref_ys
    elif ref == "ref1":
        y_ref = ref_y_1
    elif ref == "ref2":
        y_ref = ref_y_2
    else:
        y_ref = np.array([[ref]])
        
    y_ref = scaler.transform(np.array(y_ref))
    u_prev = scalerU.transform(np.array([[u_prev]]))
    y_scaled = scaler.transform(np.array([[y]]))
    
    Q = dense.quad_form(A_d, B_d, C_d, D_d, Qy, Qu, N)                      
    c = dense.lin_form(A_d, B_d, C_d, D_d, Qy, Qu, N, y_scaled, y_ref, u_prev)  
      
    F = dense.constraint_matrix_no(B, N)                   
    g = dense.upper_bound_no(umax, umin, N)
    
    U = cp.Variable((N, 1))
    
    objective = cp.Minimize(cp.quad_form(U, Q) + c @ U)
    constraints = [F @ U <= g]
    
    problem_mpc = cp.Problem(objective, constraints)
    problem_mpc.solve()
    
    return scalerU.inverse_transform(U.value[0, :].reshape(-1, 1))

def load_scalers():
    global scaler, scalerU
    scaler = joblib.load('../data/scaler.pkl')
    scalerU = joblib.load('../data/scalerU.pkl')

def load_ref():
    global ref_xs, ref_ys, ref_x_1, ref_y_1, ref_x_2, ref_y_2
    ref_xs = np.load('../data/ref_xs.npy')
    ref_ys = np.load('../data/ref_ys.npy')
    ref_x_1 = np.load('../data/ref_x_1.npy')
    ref_y_1 = np.load('../data/ref_y_1.npy')
    ref_x_2 = np.load('../data/ref_x_2.npy')
    ref_y_2 = np.load('../data/ref_y_2.npy')

def load_matrices():
    global Qy, Qu, Qx, N, umax, umin, ymax, ymin, A, B, C, D, A_d, B_d, C_d, D_d, A_noC, B_noC
    Qy = np.array([[0.01]]) #scaler.transform(20+ys)# np.array([[20]])  # Quadratic term
    Qu = np.array([[0.1]]) #scalerU.transform(1+us)# np.array([[1]])  # Quadratic term
    N = 20  # Horizon length

    umax = scalerU.transform([[100]])
    umin = scalerU.transform([[20]])
    ymax = scaler.transform([[80]]).reshape(1,-1).T
    ymin = scaler.transform([[10]]).reshape(1,-1).T
    
    A = np.load('../data/A_wC.npy')
    B = np.load('../data/B_wC.npy')
    C = np.load('../data/C_wC.npy')
    D = np.zeros((C.shape[0], B.shape[1]))
    
    A_noC = np.load('../data/A_noC.npy')
    B_noC = np.load('../data/B_noC.npy')
    
    eigenvalues, eigenvectors = np.linalg.eig(A_noC)
    dominant_indices = np.argsort(-np.abs(eigenvalues))
    nx_koopman = A_noC.shape[0]
    
    Qx = np.zeros((nx_koopman,nx_koopman))
    for i in range(nx_koopman):
        if dominant_indices[i] == 0:
            Qx[i,i] = 100
        elif dominant_indices[i] == 5:
            Qx[i,i] = 100
        elif dominant_indices[i] == 9:
            Qx[i,i] = 1
        else:
            Qx[i,i] = 1
    
    # Given parameters
    gain = 0.9669039581491512    # Example gain
    tau = 69  # Example time constant
    T = 1    # Sampling time (choose based on your application)

    # Continuous-time state-space matrices
    A_c = np.array([[-1/tau]])
    B_c = np.array([[gain/tau]])
    C_c = np.array([[1]])
    D_c = np.array([[0]])

    # Discretize using cont2discrete
    system = (A_c, B_c, C_c, D_c)
    discrete_system = cont2discrete(system, T, method='zoh')

    # Extract discrete-time matrices
    A_d, B_d, C_d, D_d, _ = discrete_system


def init():
    load_problems()
    load_scalers()
    load_matrices()
    load_ref()
    return "sucess"