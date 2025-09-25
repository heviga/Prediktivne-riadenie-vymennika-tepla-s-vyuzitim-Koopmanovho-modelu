import os
import joblib
import numpy as np
import torch
import torch.nn as nn

# NeuroMANCER imports to reconstruct the graph similarly to koopman_mpc.py
from neuromancer.system import Node, System
from neuromancer.problem import Problem
from neuromancer.modules import blocks
from neuromancer.loss import PenaltyLoss
from neuromancer import variable


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
    nx_koopman = 80
    layers = [60, 120, 180]
    ny = 1
    nsteps = 80   
    nu = 1 
    global problem, f_u, K
    
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
    
    f_y_inv = blocks.MLP(nx_koopman, ny, bias=True,
                linear_map=torch.nn.Linear,
                nonlin=torch.nn.ELU,
                hsizes=layers[::-1])
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

   # problem.load_state_dict(torch.load('./data/model_baseline.pth'),strict=False)

    import os
    model_path = os.path.join(os.path.dirname(__file__), 'data', 'model_baseline.pth')
    problem.load_state_dict(torch.load(model_path),strict=False)


def get_y(x):
    y_ = problem.nodes[4]({"x": torch.from_numpy(x).float()})
    return y_["yhat"][0].detach().numpy().reshape(1,-1).T

def get_x(y):
    global x
    y = np.array(y)
    x = problem.nodes[0]({"Y0": torch.from_numpy(y.reshape(1,-1,1)).float()})
    x = x["x"][0].detach().numpy().reshape(-1,1)

def y_plus(u):
    global x
    u = np.array(u).reshape(-1,1)
    x_plus = A@x + B@u
    y_plus = problem.nodes[4]({"x": torch.from_numpy(x_plus.reshape(1,-1)).float()})
    x = x_plus
    return y_plus["yhat"][0].detach().numpy().reshape(1,-1).T

def init():
    global A, B
    load_problems()
    A = K.weight.detach().numpy()
    B = f_u.weight.detach().numpy()
    

