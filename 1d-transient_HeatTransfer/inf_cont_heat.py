import sys
import json
import os
import tensorflow as tf
import numpy as np
# import tensorflow_probability as tfp

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)

# LOCAL IMPORTS

eqnPath = "."
sys.path.append(eqnPath)
sys.path.append("utils")
from utils.logger import Logger
from utils.neuralnetwork import NeuralNetwork
from heatutil import prep_data, plot_inf_cont_results

# HYPER PARAMETERS

if len(sys.argv) > 1:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    hp = {}
    # Data size on the solution T
    hp["N_T"] = 100
    # Collocation points size, where we’ll check for f = 0
    hp["N_f"] = 10000
    # DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width, 1-sized output [T]
    hp["layers"] = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
    hp["tf_epochs"] = 100
    hp["tf_lr"] = 0.03
    hp["tf_b1"] = 0.9
    hp["tf_eps"] = None
    # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
    hp["nt_epochs"] = 200
    hp["nt_lr"] = 0.8
    hp["nt_ncorr"] = 50
    hp["log_frequency"] = 10

# %% DEFINING THE MODEL


class TransientInformedNN(NeuralNetwork):
    def __init__(self, hp, logger, X_f, ub, lb):
        super().__init__(hp, logger, ub, lb)

        # Separating the collocation coordinates
        self.x_f = self.tensor(X_f[:, 0:1])
        self.t_f = self.tensor(X_f[:, 1:2])

    # Defining custom loss
    def loss(self, T, T_pred):
        f_pred = self.f_model()
        return tf.reduce_mean(tf.square(T - T_pred)) + \
            tf.reduce_mean(tf.square(f_pred))

    # The actual PINN
    def f_model(self):
        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_f)
            tape.watch(self.t_f)
            # Packing together the inputs
            X_f = tf.stack([self.x_f[:, 0], self.t_f[:, 0]], axis=1)

            # Getting the prediction
            T = self.model(X_f)
            # Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
            T_x = tape.gradient(T, self.x_f)

        # Getting the other derivatives
        T_xx = tape.gradient(T_x, self.x_f)
        T_t = tape.gradient(T, self.t_f)

        # Letting the tape go
        del tape

        nu = self.get_params(numpy=True)

        # Buidling the PINNs
        return T_xx-T_t


    def predict(self, X_star):
        u_star = self.model(X_star)
        f_star = self.f_model()
        return u_star.numpy(), f_star.numpy()

# %% TRAINING THE MODEL


# Getting the data
path = os.path.join(eqnPath, "data", "1d_transient_100.mat")
x, t, X, T, Exact_T, X_star, T_star, \
    X_T_train, T_train, X_f, ub, lb = prep_data(
        path, hp["N_T"], hp["N_f"], noise=0.0)

# Creating the model
logger = Logger(hp)
pinn = TransientInformedNN(hp, logger, X_f, ub, lb)

# Defining the error function for the logger and training
def error():
    T_pred, _ = pinn.predict(X_star)
    return np.linalg.norm(T_star - T_pred, 2) / np.linalg.norm(T_star, 2)


logger.set_error_fn(error)
pinn.fit(X_T_train, T_train)

# Getting the model predictions
u_pred, _ = pinn.predict(X_star)

# %% PLOTTING
plot_inf_cont_results(X_star, u_pred.flatten(), X_T_train, T_train,
                      Exact_T, X, T, x, t, save_path=eqnPath, save_hp=hp)
