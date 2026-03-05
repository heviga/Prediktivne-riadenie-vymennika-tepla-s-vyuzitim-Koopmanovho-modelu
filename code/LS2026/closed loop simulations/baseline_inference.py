import os
import joblib
import numpy as np
import torch
import torch.nn as nn

from neuromancer.system import Node, System
from neuromancer.problem import Problem
from neuromancer.modules import blocks
from neuromancer.loss import PenaltyLoss
from neuromancer import variable


problem = None
f_u = None
K = None
A = None
B = None
x = None


class Koopman_control(nn.Module):

    def __init__(self, K):
        super().__init__()
        self.K = K

    def forward(self, x, u):
        return self.K(x) + u


def load_problems():

    global problem, f_u, K

    nx_koopman = 10
    layers = [8, 16, 24, 10]
    ny = 1
    nsteps = 80
    nu = 1

    f_y = blocks.MLP(
        ny,
        nx_koopman,
        bias=True,
        linear_map=nn.Linear,
        nonlin=nn.ReLU,
        hsizes=layers,
    )

    encode_Y0 = Node(f_y, ['Y0'], ['x'], name='encoder_Y0')
    encode_Y = Node(f_y, ['Y'], ['x_latent'], name='encoder_Y')

    f_u = nn.Linear(nu, nx_koopman, bias=False)
    encode_U = Node(f_u, ['U'], ['u_latent'], name='encoder_U')

    f_y_inv = blocks.MLP(
        nx_koopman,
        ny,
        bias=True,
        linear_map=nn.Linear,
        nonlin=nn.ELU,
        hsizes=[24, 16, 8]   # ✅ presne ako checkpoint
    )

    decode_y = Node(f_y_inv, ['x'], ['yhat'], name='decoder_y')

    K = nn.Linear(nx_koopman, nx_koopman, bias=False)
    Koopman = Node(Koopman_control(K), ['x', 'u_latent'], ['x'], name='K')

    dynamics_model = System([Koopman], name='Koopman', nsteps=nsteps)

    nodes = [encode_Y0, encode_Y, encode_U, dynamics_model, decode_y]

    # ---- VARIABLES ----
    Y = variable("Y")
    yhat = variable("yhat")
    x_latent = variable("x_latent")
    u_latent = variable("u_latent")
    x_var = variable("x")

    xu_latent = x_latent + u_latent

    # ---- LOSS (presne ako tréning) ----
    y_loss = 10. * (yhat[:, 1:-1, :] == Y[:, 1:, :]) ^ 2
    y_loss.name = "y_loss"

    onestep_loss = 1. * (yhat[:, 1, :] == Y[:, 1, :]) ^ 2
    onestep_loss.name = "onestep_loss"

    x_loss = 1. * (x_var[:, 1:-1, :] == xu_latent[:, 1:, :]) ^ 2
    x_loss.name = "x_loss"

    loss = PenaltyLoss([y_loss, x_loss, onestep_loss], constraints=[])

    problem = Problem(nodes, loss)


def get_x(y):
    global x
    y = np.array(y)
    x_dict = problem.nodes[0]({
        "Y0": torch.from_numpy(y.reshape(1, -1, 1)).float()
    })
    x = x_dict["x"][0].detach().numpy().reshape(-1, 1)


def y_plus(u):
    global x
    u = np.array(u).reshape(-1, 1)

    x_plus = A @ x + B @ u

    y_dict = problem.nodes[4]({
        "x": torch.from_numpy(x_plus.reshape(1, -1)).float()
    })

    x = x_plus
    return y_dict["yhat"][0].detach().numpy().reshape(1, -1).T


def init():
    global A, B

    load_problems()

    base_dir = os.path.dirname(os.path.dirname(__file__))

    model_path = os.path.join(
        base_dir,
        "data",
        "model_20260305_115547"
    )

    # ✅ doplň príponu, ak chýba
    if not model_path.endswith(".pth"):
        model_path = model_path + ".pth"

    problem.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu")),
        strict=False
    )

    A = K.weight.detach().numpy()
    B = f_u.weight.detach().numpy()