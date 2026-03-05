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
from neuromancer.modules import blocks
from models import *

torch.manual_seed(0)


class Koopman_control(nn.Module):
    """
    Baseline class for Koopman control model
    Implements discrete-time dynamical system:
        x_k+1 = K (x_k + u_k)
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
        :param u: (torch.Tensor, shape=[batchsize, nu])
        :return: (torch.Tensor, shape=[batchsize, nx])
        """
        x = self.K(x) + u
        return x


def stack(a, b):
    return torch.cat((a, b), dim=-1)


def create_problem(ny, nu, nx_koopman, layers, lstm_features, nsteps, time, stable):
    f_yu = blocks.LSTMBlock(ny + nu, lstm_features)

    def f_yuT(data):
        s = data.shape
        res = f_yu(data.reshape(s[0] * nsteps, time, ny + nu))
        return res.reshape(s[0], nsteps, lstm_features)

    extract_time_features = Node(
        f_yuT, ["timeYU"], ["time_enc"], name="extract_time_features"
    )

    extract_time_features_0 = Node(
        f_yu, ["timeYU0"], ["time_enc0"], name="extract_time_features_0"
    )

    # instantiate output encoder neural net f_y
    f_y = blocks.MLP(
        ny + lstm_features,
        nx_koopman,
        bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ReLU,
        hsizes=layers,
    )
    # initial condition encoder
    encode_Y0 = Node(f_y, ["Y0_te"], ["x"], name="encoder_Y0")
    # observed trajectory encoder
    encode_Y = Node(f_y, ["Y_te"], ["x_latent"], name="encoder_Y")

    stack_Y0 = Node(stack, ["Y0", "time_enc0"], ["Y0_te"], name="stack_Y0")
    stack_Y = Node(stack, ["Y", "time_enc"], ["Y_te"], name="stack_Y")

    K_B = torch.nn.Linear(nu, nx_koopman, bias=False)
    encode_U = Node(K_B, ["U"], ["u_real"], name="encoder_U")
    f_y_inv = torch.nn.Linear(nx_koopman, ny, bias=False)
    # predicted trajectory decoder
    decode_y = Node(f_y_inv, ["x"], ["yhat"], name="decoder_y")

    # instantiate Koopman operator matrix
    
    if stable:
        # SVD factorized Koopman operator with bounded eigenvalues: sigma_min <= \lambda_i <= sigma_max
        K = slim.linear.SVDLinear(
            nx_koopman, nx_koopman, sigma_min=0.01, sigma_max=1.0, bias=False
        )
        # SVD penalty variable
        K_reg_error = variable(K.reg_error())
        # SVD penalty loss term
        K_reg_loss = 1.0 * (K_reg_error == 0.0)
        K_reg_loss.name = "SVD_loss"
    else:
        # linear Koopman operator without guaranteed stability
        K = torch.nn.Linear(nx_koopman, nx_koopman, bias=False)
        # symbolic Koopman model with control inputs
    Koopman = Node(Koopman_control(K), ["x", "u_real"], ["x"], name="K")

    # latent Koopmann rollout
    dynamics_model = System([Koopman], name="Koopman", nsteps=nsteps)

    # put all nodes of the Koopman model together in a list of nodes
    nodes = [
        extract_time_features,
        extract_time_features_0,
        stack_Y0,
        stack_Y,
        encode_Y0,
        encode_Y,
        encode_U,
        dynamics_model,
        decode_y,
    ]

    # variables
    Y = variable("Y")  # observed
    yhat = variable("yhat")  # predicted output
    x_latent = variable("x_latent")  # encoded output trajectory in the latent space
    # u_latent = variable('u_latent')  # encoded input trajectory in the latent space
    u_real = variable("u_real")  # real input trajectory
    x = variable("x")  # Koopman latent space trajectory

    # xu_latent = x_latent + u_latent  # latent state trajectory

    # output trajectory tracking loss
    y_loss = 10.0 * (yhat[:, 1:-1, :] == Y[:, 1:, :]) ^ 2
    y_loss.name = "y_loss"

    # one-step tracking loss
    onestep_loss = 1.0 * (yhat[:, 1, :] == Y[:, 1, :]) ^ 2
    onestep_loss.name = "onestep_loss"

    # latent trajectory tracking loss
    x_loss = 1.0 * (x[:, 1:-1, :] == x_latent[:, 1:, :]) ^ 2
    x_loss.name = "x_loss"

    # aggregate list of objective terms and constraints
    objectives = [y_loss, x_loss, onestep_loss]

    if stable:
        objectives.append(K_reg_loss)

    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints=[])

    # construct constrained optimization problem
    problem = Problem(nodes, loss)

    return problem


# function for converting np.array data to lstm input as moving window sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
    return np.array(X)


def get_data(ny, nu, nsim, nsteps, time, bs, scaler, scalerU):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.system)
    :param ts: (float) step size
    :param bs: (int) batch size

    """

    #train_sim, dev_sim, test_sim = [sys.simulate(nsim=nsim, ts=ts) for i in range(3)]
    #shift(train_sim,20)
    #shift(dev_sim,20)
    #shift(test_sim,20)
    
    with open('data/train_sim.pkl', 'rb') as fp:
        train_sim = pickle.load(fp)
    
    with open('data/val_sim.pkl', 'rb') as fp:
        dev_sim = pickle.load(fp)
    
    with open('data/test_sim.pkl', 'rb') as fp:
        test_sim = pickle.load(fp)
    
    nbatch = (nsim-time)//nsteps
    length = (nsim//nsteps) * nsteps
        # Fit scaler on training data

    # Transform both train, test, and unknown data

    trainX_b = scaler.transform(train_sim['Y'][:length])
    
    trainX = trainX_b[time:].reshape(nbatch, nsteps, ny)
    trainX = torch.tensor(trainX, dtype=torch.float32)
    trainU_b = scalerU.transform(train_sim['U'][:length])
    trainU = trainU_b[time:].reshape(nbatch, nsteps, nu)
    trainU = torch.tensor(trainU, dtype=torch.float32)
    trainUX_b = np.concatenate((trainX_b, trainU_b), axis=1)
    trainT = create_sequences(trainUX_b, time)
    trainT = trainT.reshape(nbatch, nsteps, time, nu+ny)
    trainT = torch.tensor(trainT, dtype=torch.float32)
    
    exp_train_data = {'Y': trainX, 'Y0': trainX[:, 0:1, :],
                            'U': trainU, 'timeYU0': trainT[:, 0:1, :, :].reshape(nbatch, time, nu+ny),
                            'timeYU': trainT}
    
    train_data = DictDataset({'Y': trainX, 'Y0': trainX[:, 0:1, :],
                              'U': trainU, 'timeYU0': trainT[:, 0:1, :, :].reshape(nbatch, time, nu+ny),
                              'timeYU': trainT}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                              collate_fn=train_data.collate_fn, shuffle=True)
    
    devX_b = scaler.transform(dev_sim['Y'][:length])
    
    devX = devX_b[time:].reshape(nbatch, nsteps, ny)
    devX = torch.tensor(devX, dtype=torch.float32)
    devU_b = scalerU.transform(dev_sim['U'][:length])
    devU = devU_b[time:].reshape(nbatch, nsteps, nu)
    devU = torch.tensor(devU, dtype=torch.float32)
    devUX_b = np.concatenate((devX_b, devU_b), axis=1)
    devT = create_sequences(devUX_b, time)
    devT = devT.reshape(nbatch, nsteps, time, nu+ny)
    devT = torch.tensor(devT, dtype=torch.float32)
    
    dev_data = DictDataset({'Y': devX, 'Y0': devX[:, 0:1, :],
                              'U': devU, 'timeYU0': devT[:, 0:1, :, :].reshape(nbatch, time, nu+ny),
                              'timeYU': devT}, name='dev')
    dev_loader = DataLoader(dev_data, batch_size=bs,
                              collate_fn=dev_data.collate_fn, shuffle=True)


    testX_b = scaler.transform(test_sim['Y'][:length])
    
    testX = testX_b[time:].reshape(nbatch, nsteps, ny)
    testX = torch.tensor(testX, dtype=torch.float32)
    testU_b = scalerU.transform(test_sim['U'][:length])
    testU = testU_b[time:].reshape(nbatch, nsteps, nu)
    testU = torch.tensor(testU, dtype=torch.float32)
    testUX_b = np.concatenate((testX_b, testU_b), axis=1)
    testT = create_sequences(testUX_b, time)
    testT = testT.reshape(nbatch, nsteps, time, nu+ny)
    testT = torch.tensor(testT, dtype=torch.float32)
    
    test_data = {'Y': testX, 'Y0': testX[:, 0:1, :],
                              'U': testU, 'timeYU0': testT[:, 0:1, :, :].reshape(nbatch, time, nu+ny),
                              'timeYU': testT}

    return train_loader, dev_loader, test_data, exp_train_data