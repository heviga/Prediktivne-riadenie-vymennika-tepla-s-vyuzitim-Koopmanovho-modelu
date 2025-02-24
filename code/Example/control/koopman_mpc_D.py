import sys
sys.path.append('../../functions')
 menil si modely musis to upravit
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

def load_problem():
    nx_koopman = 20
    layers = [32, 32]
    ny = 1
    nsteps = 40   
    nu = 1 
    global problem
    
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
    
    # instantiate input encoder net f_u for matrix D
    f_u_D = torch.nn.Linear(nu, ny, bias=False)
    # initial condition encoder
    encode_U_D = Node(f_u_D, ['U'], ['u_final'], name='encoder_U_D')

    f_y_inv = torch.nn.Linear(nx_koopman, ny, bias=False)
    # predicted trajectory decoder
    decode_y = Node(f_y_inv, ['x'], ['yhat'], name='decoder_y')
    
    K = torch.nn.Linear(nx_koopman, nx_koopman, bias=False)
    # symbolic Koopman model with control inputs
    Koopman = Node(Koopman_control(K), ['x', 'u_latent'], ['x'], name='K')

    # latent Koopmann rollout
    dynamics_model = System([Koopman], name='Koopman', nsteps=nsteps)
    
    nodes = [encode_Y0, encode_Y, encode_U, dynamics_model, decode_y]
    
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
    problem.load_state_dict(torch.load('../data/model_20241121_085038.pth'),strict=False)

def generate_u(y):
    return 50

def get_x(y):
    y = np.array([[y]])
    x = problem.nodes[0]({"Y0": torch.from_numpy(y).float()})
    return x["x"][0].detach().numpy().reshape(1,-1).T

def get_koopman_u(y):
    y_scaled = scaler.transform(np.array([[y]]))
    x0 = get_x(y_scaled)
    Q = dense.quad_form_or(A, B, C, D, Qy, Qu, N)                      
    c = dense.lin_form_or(A, B, C, D, Qy, N, x0)    
    F = dense.constraint_matrix_or(A,B, C, D, N)                   
    g = dense.upper_bound_or(A, C, N, x0, ymax, ymin, umax, umin)
    
    U = cp.Variable((N, 1))
    
    objective = cp.Minimize(cp.quad_form(U, Q) + c @ U)
    constraints = [F @ U <= g]
    
    problem_mpc = cp.Problem(objective, constraints)
    problem_mpc.solve()
    
    return scalerU.inverse_transform(U.value[0, :].reshape(-1, 1))
    
def get_strejc_u(y):
    y_scaled = scaler.transform(np.array([[y]]))
    Q = dense.quad_form_or(A_d, B_d, C_d, D_d, Qy, Qu, N)                      
    c = dense.lin_form_or(A_d, B_d, C_d, D_d, Qy, N, y_scaled)    
    F = dense.constraint_matrix_or(A_d, B_d, C_d, D_d, N)                   
    g = dense.upper_bound_or(A_d, C_d, N, y_scaled, ymax, ymin, umax, umin)
    
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



def load_matrices():
    global Qy, Qu, N, umax, umin, ymax, ymin, A, B, C, D, A_d, B_d, C_d, D_d
    Qy = np.array([[10]]) #scaler.transform(20+ys)# np.array([[20]])  # Quadratic term
    Qu = np.array([[1]]) #scalerU.transform(1+us)# np.array([[1]])  # Quadratic term
    N = 20  # Horizon length

    umax = scalerU.transform([[100]])
    umin = scalerU.transform([[20]])
    ymax = scaler.transform([[80]]).reshape(1,-1).T
    ymin = scaler.transform([[10]]).reshape(1,-1).T
    
    A = np.load('../data/A_DeReK_False.npy')
    B = np.load('../data/B_DeReK_False.npy')
    C = np.load('../data/C_DeReK_False.npy')
    D = np.zeros((C.shape[0], B.shape[1]))
    
    # Given parameters
    gain = 0.991164078706561    # Example gain
    tau = 61  # Example time constant
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
    load_problem()
    load_scalers()
    load_matrices()
    return "sucess"