# import os
# import joblib
# import numpy as np
# import torch
# import torch.nn as nn

# from neuromancer.system import Node, System
# from neuromancer.problem import Problem
# from neuromancer.modules import blocks
# from neuromancer.loss import PenaltyLoss
# from neuromancer import variable


# problem = None
# f_u = None
# K = None
# A = None
# B = None
# x = None


# class Koopman_control(nn.Module):

#     def __init__(self, K):
#         super().__init__()
#         self.K = K

#     def forward(self, x, u):
#         return self.K(x) + u


# def load_problems():

#     global problem, f_u, K

#     nx_koopman = 80
#     layers = [8, 16, 24]
#     ny = 1
#     nsteps = 80
#     nu = 1


#     f_y = blocks.MLP(
#         ny,
#         nx_koopman,
#         bias=True,
#         linear_map=nn.Linear,
#         nonlin=nn.ReLU,
#         hsizes=layers,
#     )

#     encode_Y0 = Node(f_y, ['Y0'], ['x'], name='encoder_Y0')
#     encode_Y = Node(f_y, ['Y'], ['x_latent'], name='encoder_Y')

#     f_u = nn.Linear(nu, nx_koopman, bias=False)
#     encode_U = Node(f_u, ['U'], ['u_latent'], name='encoder_U')

#     f_y_inv = blocks.MLP(
#         nx_koopman,
#         ny,
#         bias=True,
#         linear_map=nn.Linear,
#         nonlin=nn.ELU,
#         hsizes=[24, 16, 8]   # ✅ presne ako checkpoint
#     )

#     decode_y = Node(f_y_inv, ['x'], ['yhat'], name='decoder_y')

#     K = nn.Linear(nx_koopman, nx_koopman, bias=False)
#     Koopman = Node(Koopman_control(K), ['x', 'u_latent'], ['x'], name='K')

#     dynamics_model = System([Koopman], name='Koopman', nsteps=nsteps)

#     nodes = [encode_Y0, encode_Y, encode_U, dynamics_model, decode_y]

#     # ---- VARIABLES ----
#     Y = variable("Y")
#     yhat = variable("yhat")
#     x_latent = variable("x_latent")
#     u_latent = variable("u_latent")
#     x_var = variable("x")

#     xu_latent = x_latent + u_latent

#     # ---- LOSS (presne ako tréning) ----
#     y_loss = 10. * (yhat[:, 1:-1, :] == Y[:, 1:, :]) ^ 2
#     y_loss.name = "y_loss"

#     onestep_loss = 1. * (yhat[:, 1, :] == Y[:, 1, :]) ^ 2
#     onestep_loss.name = "onestep_loss"

#     x_loss = 1. * (x_var[:, 1:-1, :] == xu_latent[:, 1:, :]) ^ 2
#     x_loss.name = "x_loss"

#     loss = PenaltyLoss([y_loss, x_loss, onestep_loss], constraints=[])

#     problem = Problem(nodes, loss)


# def get_x(y):
#     global x
#     y = np.array(y)
#     x_dict = problem.nodes[0]({
#         "Y0": torch.from_numpy(y.reshape(1, -1, 1)).float()
#     })
#     x = x_dict["x"][0].detach().numpy().reshape(-1, 1)


# def y_plus(u):
#     global x
#     u = np.array(u).reshape(-1, 1)

#     x_plus = A @ x + B @ u

#     y_dict = problem.nodes[4]({
#         "x": torch.from_numpy(x_plus.reshape(1, -1)).float()
#     })

#     x = x_plus
#     return y_dict["yhat"][0].detach().numpy().reshape(1, -1).T


# def init():
#     global A, B

#     load_problems()

#     base_dir = os.path.dirname(os.path.dirname(__file__))

#     model_path = os.path.join(
#         base_dir,
#         "data",
#         "model_20260305_171851"
#     )

#     # ✅ doplň príponu, ak chýba
#     if not model_path.endswith(".pth"):
#         model_path = model_path + ".pth"

#     problem.load_state_dict(
#         torch.load(model_path, map_location=torch.device("cpu")),
#         strict=False
#     )

#     A = K.weight.detach().numpy()
#     B = f_u.weight.detach().numpy()

# # baseline_inference.py (SIMPLE)

# import os
# import numpy as np
# import torch
# import torch.nn as nn

# from neuromancer.system import Node, System
# from neuromancer.problem import Problem
# from neuromancer.modules import blocks
# from neuromancer.loss import PenaltyLoss
# from neuromancer import variable

# problem = None
# f_u = None
# K = None
# A = None
# B = None
# x = None


# class Koopman_control(nn.Module):
#     def __init__(self, K):
#         super().__init__()
#         self.K = K

#     def forward(self, x, u):
#         return self.K(x) + u


# def load_problems():
#     global problem, f_u, K

#     nx_koopman = 80
#     layers = [8, 16, 24,10]
#     ny = 1
#     nsteps = 50   # ✅ match training (you used nsteps=50 in get_data)
#     nu = 1

#     # encoder
#     f_y = blocks.MLP(
#         ny, nx_koopman,
#         bias=True,
#         linear_map=nn.Linear,
#         nonlin=nn.ReLU,
#         hsizes=layers,
#     )
#     encode_Y0 = Node(f_y, ['Y0'], ['x'], name='encoder_Y0')
#     encode_Y  = Node(f_y, ['Y'],  ['x_latent'], name='encoder_Y')

#     # input encoder
#     f_u = nn.Linear(nu, nx_koopman, bias=False)
#     encode_U = Node(f_u, ['U'], ['u_latent'], name='encoder_U')

#     # decoder (must match checkpoint)
#     f_y_inv = blocks.MLP(
#         nx_koopman, ny,
#         bias=True,
#         linear_map=nn.Linear,
#         nonlin=nn.ELU,
#         hsizes=[24, 16, 8],
#     )
#     decode_y = Node(f_y_inv, ['x'], ['yhat'], name='decoder_y')

#     # koopman operator
#     K = nn.Linear(nx_koopman, nx_koopman, bias=False)
#     Koopman = Node(Koopman_control(K), ['x', 'u_latent'], ['x'], name='K')
#     dynamics_model = System([Koopman], name='Koopman', nsteps=nsteps)

#     nodes = [encode_Y0, encode_Y, encode_U, dynamics_model, decode_y]

#     # loss (doesn't matter much for inference, but keep consistent)
#     Y        = variable("Y")
#     yhat     = variable("yhat")
#     x_latent = variable("x_latent")
#     u_latent = variable("u_latent")
#     x_var    = variable("x")

#     xu_latent = x_latent + u_latent

#     y_loss = 10. * (yhat[:, 1:-1, :] == Y[:, 1:, :]) ^ 2
#     y_loss.name = "y_loss"

#     onestep_loss = 1. * (yhat[:, 1, :] == Y[:, 1, :]) ^ 2
#     onestep_loss.name = "onestep_loss"

#     x_loss = 1. * (x_var[:, 1:-1, :] == xu_latent[:, 1:, :]) ^ 2
#     x_loss.name = "x_loss"

#     loss = PenaltyLoss([y_loss, x_loss, onestep_loss], constraints=[])
#     problem = Problem(nodes, loss)


# def init(model_name="model_20260305_115547"):
#     global A, B

#     load_problems()

#     base_dir = os.path.dirname(os.path.dirname(__file__))  # .../LS2026
#     model_path = os.path.join(base_dir, "data", model_name)

#     # add suffix if missing
#     if not model_path.endswith(".pth"):
#         model_path += ".pth"

#     print("Loaded model:", model_path)

#     problem.load_state_dict(
#         torch.load(model_path, map_location=torch.device("cpu")),
#         strict=False
#     )

#     A = K.weight.detach().numpy()
#     B = f_u.weight.detach().numpy()


# def get_x(y):
#     """y is already SCALED"""
#     global x
#     y = np.array(y, dtype=float)
#     out = problem.nodes[0]({"Y0": torch.from_numpy(y.reshape(1, -1, 1)).float()})
#     x = out["x"][0].detach().numpy().reshape(-1, 1)


# def y_plus(u):
#     global x
#     u = np.array(u, dtype=float).reshape(1, 1)   # [1,nu]

#     # u_latent = f_u(u)
#     u_lat = problem.nodes[2]({"U": torch.from_numpy(u.reshape(1,1,1)).float()})["u_latent"][0].detach().numpy().reshape(-1,1)

#     x_plus = A @ x + u_lat

#     y_dict = problem.nodes[4]({"x": torch.from_numpy(x_plus.reshape(1, -1)).float()})
#     x = x_plus
#     return y_dict["yhat"][0].detach().numpy().reshape(1, -1).T

# def step(u, y_meas=None):
#     """
#     One simulation step with optional measurement correction.
#     - u: scaled
#     - y_meas: scaled (if provided, we re-encode latent from measurement first)
#     """
#     if y_meas is not None:
#         get_x(y_meas)
#     return y_plus(u)


# baseline_inference.py  (ROBUST + SIMPLE)
# baseline_inference.py
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from neuromancer.modules import blocks

# f_y = None
# f_y_inv = None
# f_u = None
# K = None
# A = None
# B = None
# x = None
# _loaded = False

# LAYERS = [80, 160, 240]
# NY = 1
# NU = 1
# NX_KOOPMAN = 240


# def init(model_name="model_baseline1"):
#     global f_y, f_y_inv, f_u, K, A, B, x, _loaded

#     base_dir = os.path.dirname(os.path.dirname(__file__))
#     model_path = os.path.join(base_dir, "data", model_name)
#     if not model_path.endswith(".pth"):
#         model_path += ".pth"

#     ckpt = torch.load(model_path, map_location=torch.device("cpu"))

#     f_y = blocks.MLP(
#         NY, NX_KOOPMAN,
#         bias=True,
#         linear_map=nn.Linear,
#         nonlin=nn.ReLU,
#         hsizes=LAYERS
#     )

#     f_u = nn.Linear(NU, NX_KOOPMAN, bias=False)

#     f_y_inv = blocks.MLP(
#         NX_KOOPMAN, NY,
#         bias=True,
#         linear_map=nn.Linear,
#         nonlin=nn.ELU,
#         hsizes=[240, 160, 80]
#     )

#     K = nn.Linear(NX_KOOPMAN, NX_KOOPMAN, bias=False)

#     f_y.load_state_dict({k.replace("nodes.0.callable.", ""): v for k, v in ckpt.items() if k.startswith("nodes.0.callable.")}, strict=True)
#     f_u.load_state_dict({k.replace("nodes.2.callable.", ""): v for k, v in ckpt.items() if k.startswith("nodes.2.callable.")}, strict=True)
#     K.load_state_dict({k.replace("nodes.3.nodes.0.callable.K.", ""): v for k, v in ckpt.items() if k.startswith("nodes.3.nodes.0.callable.K.")}, strict=True)
#     f_y_inv.load_state_dict({k.replace("nodes.4.callable.", ""): v for k, v in ckpt.items() if k.startswith("nodes.4.callable.")}, strict=True)

#     f_y.eval()
#     f_u.eval()
#     f_y_inv.eval()
#     K.eval()

#     A = K.weight.detach().cpu().numpy()
#     B = f_u.weight.detach().cpu().numpy()
#     x = None
#     _loaded = True


# def reset(y_scaled):
#     global x
#     y = np.array([[float(y_scaled)]], dtype=np.float32)
#     with torch.no_grad():
#         xt = f_y(torch.from_numpy(y))
#     x = xt.cpu().numpy().reshape(-1, 1)
#     return x


# def step(u_scaled):
#     global x
#     u = np.array([[float(u_scaled)]], dtype=np.float32)

#     with torch.no_grad():
#         u_lat = f_u(torch.from_numpy(u)).cpu().numpy().reshape(-1, 1)

#     x = A @ x + u_lat

#     with torch.no_grad():
#         yhat = f_y_inv(torch.from_numpy(x.reshape(1, -1)).float()).cpu().numpy()

#     return float(yhat[0, 0])
import os
import sys
import types

# Prevent wandb from actually loading (it crashes in MATLAB's Python host)
_fake_wandb = types.ModuleType('wandb')
_fake_wandb.sdk = types.ModuleType('wandb.sdk')
_fake_wandb.init = lambda *a, **kw: None
_fake_wandb.log = lambda *a, **kw: None
_fake_wandb.finish = lambda *a, **kw: None
sys.modules.setdefault('wandb', _fake_wandb)
sys.modules.setdefault('wandb.sdk', _fake_wandb.sdk)

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

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
    nx_koopman = 24
    layers = [8,16,24]
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
    
    f_y_inv = blocks.MLP(
        nx_koopman,
        ny,
        bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ELU,
        hsizes=layers[::-1]
    )
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
    # model_path = os.path.join(os.path.dirname(__file__), 'data', 'model_baseline.pth')
    # problem.load_state_dict(torch.load(model_path),strict=False)

    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'data', 'baseline.pth')
    problem.load_state_dict(torch.load("../data/baseline.pth"),strict=False)


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
    