#import numpy as np
#import tensorflow as tf
#import matplotlib.pyplot as plt
#import csv
#import os
#from os import system
#tf.compat.v1.enable_eager_execution()
#
#
## Constants for the potential
#m = 1.0        # Mass
#Omega = 1.0    # Frequency of the harmonic trap
#A = 1.0        # Amplitude of the Gaussian potential
#sigma = 1.0    # Width of the Gaussian potential
#T = 10.0       # Final time
#dt = 0.01      # Time step
#x_star = 0.5   # Final position sought
#v_th = 0.01    # Velocity threshold
#t_values = np.linspace(1e-8, T, int(T/dt))
#
## Define the neural network model for xi(t)
#class XiModel(tf.keras.Model):
#    def __init__(self):
#        super(XiModel, self).__init__()
#        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
#        self.dense2 = tf.keras.layers.Dense(32, activation='tanh')
#        self.dense3 = tf.keras.layers.Dense(16, activation='relu')
#        self.output_layer = tf.keras.layers.Dense(1)
#        self.skip_connection = tf.keras.layers.Dense(1)  # For the skip connection
#
#    def call(self, inputs):
#        x = self.dense1(inputs)
#        x = self.dense2(x)
#        x = self.dense3(x)
#        skip = self.skip_connection(inputs)  # Compute skip connection
#        return self.output_layer(x) + skip  # Combine main path with skip
#
## Load or instantiate the model
#model_path = "xi_model.h5"
#xi_model = XiModel()
#if os.path.exists(model_path):
#    dummy_input = tf.convert_to_tensor([[0.0]], dtype=tf.float32)
#    xi_model(dummy_input)
#    xi_model.load_weights(model_path)
#    print("Model loaded from file.")
#else:
#    print("No saved model found. Starting from scratch.")
#
## Function to compute the force (negative derivative of potential)
#def force(x, xi):
#    return -(Omega**2 * (x - xi)) - (2.0 * A * (x - xi) / sigma) * tf.exp(-(x - xi)**2 / sigma)
#
## RK4 step for updating state
#def rk4_step(x, v, xi_t, dt):
#    # Derivatives for RK4 method
#    def dxdt(v):
#        return v
#    def dvdt(x, xi):
#        return force(x, xi) / m
#
#    # Compute RK4 coefficients
#    k1_x = dxdt(v)
#    k1_v = dvdt(x, xi_t)
#
#    k2_x = dxdt(v + 0.5 * dt * k1_v)
#    k2_v = dvdt(x + 0.5 * dt * k1_x, xi_t)
#
#    k3_x = dxdt(v + 0.5 * dt * k2_v)
#    k3_v = dvdt(x + 0.5 * dt * k2_x, xi_t)
#
#    k4_x = dxdt(v + dt * k3_v)
#    k4_v = dvdt(x + dt * k3_x, xi_t)
#
#    # Update x and v
#    x_new = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
#    v_new = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
#
#    return x_new, v_new
#
## Define the xi function using the neural network
#def xi(t):
#    t_input = tf.convert_to_tensor([[t]], dtype=tf.float32)  # Convert to tensor
#    return xi_model(t_input)[0, 0]  # Get the output from the model
#
## Loss function for optimization
#def loss_func():
#    MSE = 0.0
#    x, v = 1.0, 0.0
#    for t in t_values:
#        xi_t = xi(t)  # Compute xi(t) at time t, given by neural network!
#        # Perform RK4 step
#        x, v = rk4_step(x, v, xi_t, dt)
#    print(f"x = {x:.4f}, v = {v:.4f}, x_star = {x_star:.4f}, v_th = {v_th:.4f}, xi(T) = {xi(T):.4f}")
#    MSE = (x_star - x) ** 2
#    abs_v = abs(v)
#    if abs_v > v_th:
#        MSE += (abs_v - v_th) ** 2
#    diff_xi_T = x_star - xi(T)
#    MSE += diff_xi_T ** 2
#    diff_xi_0 = xi(0)
#    MSE += diff_xi_0 ** 2
#    return MSE
#
## Training loop
#learning_rate = 0.001
#optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#x_values = []   # To store x(t)
#v_values = []   # To store v(t)
#xi_values = []  # To store xi(t)
#try:
#    for epoch in range(1000):  # Number of training iterations
#        with tf.GradientTape() as tape:
#            loss_value = loss_func()  # Compute the loss
#        grads = tape.gradient(loss_value, xi_model.trainable_variables)  # Compute gradients
#        optimizer.apply_gradients(zip(grads, xi_model.trainable_variables))  # Update model weights
#        print(f'Epoch {epoch}, Loss: {loss_value.numpy()}')
#
#        if epoch % 5 == 0:
#            x_values = []   # To store x(t)
#            v_values = []   # To store v(t)
#            xi_values = []  # To store xi(t)
#            # Initial conditions
#            x = 1.0  # Initial position x(0)
#            v = 0.0  # Initial velocity v(0)
#            for t in t_values:
#                xi_t = xi(t)  # Compute xi(t) at time t
#                xi_values.append(xi_t)
#
#                # Perform RK4 step
#                x, v = rk4_step(x, v, xi_t, dt)
#
#                # Store the position x(t) and velocity v(t)
#                x_values.append(x)
#                v_values.append(v)
#            plt.plot(t_values, x_values, label='x(t) [m]', color='blue')
#            plt.plot(t_values, v_values, label='v(t) [m/s]', color='green', linestyle=':')
#            plt.plot(t_values, xi_values, label = r'$\xi(t)$', color='red', linestyle='--')
#            plt.legend()
#            plt.draw()
#            plt.pause(1)
#            plt.close()
#
#except KeyboardInterrupt:
#    print("\nTraining interrupted. Saving model and data...")
#
#    # Save the model
#    xi_model.save_weights(model_path)
#    print("Model saved.")
#
#    # Save data to CSV
#    data_path = "trajectory_data.csv"
#    with open(data_path, mode='w', newline='') as file:
#        writer = csv.writer(file)
#        writer.writerow(["t_values", "xi_values", "x_values", "v_values"])
#        for i in range(len(t_values)):
#            writer.writerow([t_values[i], xi_values[i].numpy().item(), x_values[i].numpy().item(), v_values[i].numpy().item()])
#    print("Data saved to CSV.")
#
#    x_values = []   # To store x(t)
#    v_values = []   # To store v(t)
#    xi_values = []  # To store xi(t)
#    # Initial conditions
#    x = 1.0  # Initial position x(0)
#    v = 0.0  # Initial velocity v(0)
#    for t in t_values:
#        xi_t = xi(t)  # Compute xi(t) at time t
#        xi_values.append(xi_t)
#
#        # Perform RK4 step
#        x, v = rk4_step(x, v, xi_t, dt)
#
#        # Store the position x(t) and velocity v(t)
#        x_values.append(x)
#        v_values.append(v)
#    plt.plot(t_values, x_values, label='x(t) [m]', color='blue')
#    plt.plot(t_values, v_values, label='v(t) [m/s]', color='green', linestyle=':')
#    plt.plot(t_values, xi_values, label = r'$\xi(t)$', color='red', linestyle='--')
#    plt.legend()
#    plt.savefig("trajectory_data.svg")
#    system(f"rsvg-convert -f pdf -o trajectory_data.pdf trajectory_data.svg")
#    system("rm trajectory_data.svg")
#    system("open trajectory_data.pdf")

#import numpy as np
#import tensorflow as tf
#import matplotlib.pyplot as plt
#import csv
#import os
#from os import system
#
## Enable eager execution
#tf.compat.v1.enable_eager_execution()
#
## Constants for the potential
#m = 1.0        # Mass
#Omega = 1.0    # Frequency of the harmonic trap
#A = 1.0        # Amplitude of the Gaussian potential
#sigma = 1.0    # Width of the Gaussian potential
#T = 10.0       # Final time
#dt = 0.01      # Time step
#x_star = 0.5   # Final position sought
#v_th = 0.01    # Velocity threshold
#t_values = np.linspace(1e-8, T, int(T / dt))
#
## Define the neural network for xi(t) manually
#class XiModel:
#    def __init__(self):
#        # Initialize weights and biases for each layer with reduced neurons
#        self.w1 = tf.Variable(tf.random.normal([1, 8]), dtype=tf.float32)   # Reduced to 8 neurons
#        self.b1 = tf.Variable(tf.zeros([8]), dtype=tf.float32)
#        self.w2 = tf.Variable(tf.random.normal([8, 4]), dtype=tf.float32)    # Reduced to 8 neurons
#        self.b2 = tf.Variable(tf.zeros([4]), dtype=tf.float32)
#        self.w3 = tf.Variable(tf.random.normal([4, 4]), dtype=tf.float32)     # Reduced to 4 neurons
#        self.b3 = tf.Variable(tf.zeros([4]), dtype=tf.float32)
#        self.w_out = tf.Variable(tf.random.normal([4, 1]), dtype=tf.float32)  # Output layer remains 1 neuron
#        self.b_out = tf.Variable(tf.zeros([1]), dtype=tf.float32)
#
#        # Skip connection remains the same
#        self.w_skip = tf.Variable(tf.random.normal([1, 1]), dtype=tf.float32)
#        self.b_skip = tf.Variable(tf.zeros([1]), dtype=tf.float32)
#
#    def __call__(self, inputs):
#        # Forward pass with activations and reduced neurons
#        x = tf.nn.relu(tf.matmul(inputs, self.w1) + self.b1)
#        x = tf.nn.tanh(tf.matmul(x, self.w2) + self.b2)
#        x = tf.nn.relu(tf.matmul(x, self.w3) + self.b3)
#
#        # Output combination of main path and skip connection
#        output_main = tf.matmul(x, self.w_out) + self.b_out
#        output_skip = tf.matmul(inputs, self.w_skip) + self.b_skip
#        return output_main + output_skip
#
## Instantiate the model
#model = XiModel()
#
## Load or instantiate the model
#model_path = "xi_model.npz"
#if os.path.exists(model_path):
#    npzfile = np.load(model_path)
#    for k, v in npzfile.items():
#        print(k, tf.Variable(v))
#        setattr(model, k, tf.Variable(v))
#    print("Model loaded from file.")
#else:
#    print("No saved model found. Starting from scratch.")
#
## Function to compute the force (negative derivative of potential)
#def force(x, xi):
#    return -(Omega**2 * (x - xi)) - (2.0 * A * (x - xi) / sigma) * tf.exp(-(x - xi)**2 / sigma)
#
## RK4 step for updating state
#def rk4_step(x, v, xi_t, dt):
#    def dxdt(v):
#        return v
#
#    def dvdt(x, xi):
#        return force(x, xi) / m
#
#    k1_x = dxdt(v)
#    k1_v = dvdt(x, xi_t)
#
#    k2_x = dxdt(v + 0.5 * dt * k1_v)
#    k2_v = dvdt(x + 0.5 * dt * k1_x, xi_t)
#
#    k3_x = dxdt(v + 0.5 * dt * k2_v)
#    k3_v = dvdt(x + 0.5 * dt * k2_x, xi_t)
#
#    k4_x = dxdt(v + dt * k3_v)
#    k4_v = dvdt(x + dt * k3_x, xi_t)
#
#    x_new = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
#    v_new = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
#
#    return x_new, v_new
#
## Define the xi function using the neural network
#def xi(t):
#    t_input = tf.convert_to_tensor([[t]], dtype=tf.float32)  # Convert to tensor
#    return model(t_input)[0, 0]  # Get the output from the model
#
## Loss function for optimization
#def loss_func():
#    MSE = 0.0
#    x, v = 1.0, 0.0
#    smoothness_penalty = 0.0  # Initialize smoothness penalty
#    xi_values_temp = []  # Temporary storage for xi values to calculate smoothness
#
#    for t in t_values:
#        xi_t = xi(t)  # Compute xi(t) at time t
#        xi_values_temp.append(xi_t)  # Store xi values for smoothness calculation
#
#        # Perform RK4 step
#        x, v = rk4_step(x, v, xi_t, dt)
#
#    print(f"x = {x:.4f}, v = {v:.4f}, x_star = {x_star:.4f}, v_th = {v_th:.4f}, xi(T) = {xi_values_temp[-1]:.4f}")
#
#    # Compute the smoothness penalty
#    for i in range(1, len(xi_values_temp)):
#        # Calculate the difference in xi values
#        delta_xi = xi_values_temp[i] - xi_values_temp[i - 1]
#        # Calculate the difference in t values
#        delta_t = t_values[i] - t_values[i - 1]
#
#        # Compute the derivative
#        if delta_t > 0:  # Avoid division by zero
#            derivative = delta_xi / delta_t
#            smoothness_penalty += tf.reduce_sum(tf.square(derivative))  # Penalty based on the square of the derivative
#
#    # Regular MSE calculation
#    MSE = (x_star - x) ** 2
#    abs_v = abs(v)
#    if abs_v > v_th:
#        MSE += (abs_v - v_th) ** 2
#    diff_xi_T = x_star - xi(T)
#    MSE += diff_xi_T ** 2
#    diff_xi_0 = xi(0)
#    MSE += diff_xi_0 ** 2
#
#    # Combine MSE with smoothness penalty (scale the penalty as needed)
#    total_loss = MSE + 0.005 * smoothness_penalty  # Adjust the scale factor (0.1) to tune the smoothness constraint
#    return total_loss
#
## Training loop
#learning_rate = 0.1
#optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
#x_values, v_values, xi_values = [], [], []
#plot_progress = False
#epoch = 0
#
#try:
#    while True:
#        with tf.GradientTape() as tape:
#            loss_value = loss_func()
#        grads = tape.gradient(loss_value, [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3, model.w_out, model.b_out, model.w_skip, model.b_skip])
#        optimizer.apply_gradients(zip(grads, [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3, model.w_out, model.b_out, model.w_skip, model.b_skip]))
#        print(f'Epoch {epoch}, Loss: {loss_value.numpy()}')
#
#        if plot_progress and epoch % 5 == 0:
#            x_values, v_values, xi_values = [], [], []
#            x, v = 1.0, 0.0  # Initial conditions
#            for t in t_values:
#                xi_t = xi(t)
#                xi_values.append(xi_t)
#                x, v = rk4_step(x, v, xi_t, dt)
#                x_values.append(x)
#                v_values.append(v)
#            plt.plot(t_values, x_values, label='x(t) [m]', color='blue')
#            plt.plot(t_values, v_values, label='v(t) [m/s]', color='green', linestyle=':')
#            plt.plot(t_values, xi_values, label=r'$\xi(t)$', color='red', linestyle='--')
#            plt.legend()
#            plt.draw()
#            plt.pause(1)
#            plt.close()
#        epoch += 1
#
#except KeyboardInterrupt:
#    print("\nTraining interrupted. Saving model and data...")
#
#    # Save the model
#    np.savez(model_path, w1=model.w1.numpy(), b1=model.b1.numpy(), w2=model.w2.numpy(), b2=model.b2.numpy(),
#             w3=model.w3.numpy(), b3=model.b3.numpy(), w_out=model.w_out.numpy(), b_out=model.b_out.numpy(),
#             w_skip=model.w_skip.numpy(), b_skip=model.b_skip.numpy())
#    print("Model saved.")
#
#    # Save data to CSV
#    data_path = "trajectory_data.csv"
#    x_values, v_values, xi_values = [], [], []
#    x, v = 1.0, 0.0  # Initial conditions
#    for t in t_values:
#        xi_t = xi(t)
#        xi_values.append(xi_t)
#        x, v = rk4_step(x, v, xi_t, dt)
#        x_values.append(x)
#        v_values.append(v)
#    with open(data_path, mode='w', newline='') as file:
#        writer = csv.writer(file)
#        writer.writerow(["t_values", "xi_values", "x_values", "v_values"])
#        for i in range(len(t_values)):
#            writer.writerow([t_values[i], xi_values[i].numpy().item(), x_values[i].numpy().item(), v_values[i].numpy().item()])
#    print("Data saved to CSV.")
#
#    plt.plot(t_values, x_values, label='x(t) [m]', color='blue')
#    plt.plot(t_values, v_values, label='v(t) [m/s]', color='green', linestyle=':')
#    plt.plot(t_values, xi_values, label=r'$\xi(t)$', color='red', linestyle='--')
#    plt.legend()
#    plt.savefig("trajectory_data.svg")
#    system(f"rsvg-convert -f pdf -o trajectory_data.pdf trajectory_data.svg")
#    system("rm trajectory_data.svg")
#    system("open trajectory_data.pdf")

# After training, you can evaluate or use the model further


#import numpy as np
#import gym
#from gym import spaces
#import warnings
#import sys
#from sympy.utilities.lambdify import lambdify
#from sympy import symbols
#from rl.memory import SequentialMemory
#from rl.policy import BoltzmannQPolicy
#from rl.agents import DDPGAgent
#from tensorflow.keras.layers import Dense, Flatten, Activation, Input
#from tensorflow.keras.models import Sequential
#from tensorflow_addons.optimizers import LazyAdam
#from keras.layers import Concatenate
#from tensorflow.keras import Model, initializers
#import tensorflow.keras.backend as K
#from time import time
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
#from scipy.optimize import ridder, fsolve
#from gym import Env
#from gym.spaces import Box
#from pysr import PySRRegressor
#from math import isclose
#import argparse
#import tensorflow as tf
#import torch
#import torch.nn as nn
#import torch.optim as optim
#tf.compat.v1.disable_eager_execution()
#warnings.filterwarnings("ignore")
#
#class XiNN(nn.Module):
#    def __init__(self):
#        super(XiNN, self).__init__()
#        self.fc1 = nn.Linear(1, 16)  # Input layer
#        self.fc2 = nn.Linear(16, 32)  # Hidden layer
#        self.fc3 = nn.Linear(32, 1)   # Output layer
#
#    def forward(self, x):
#        x = torch.relu(self.fc1(x))
#        x = torch.relu(self.fc2(x))
#        return self.fc3(x)
#
#class CustomEnv(gym.Env):
#    def __init__(self):
#        super(CustomEnv, self).__init__()
#
#        # Action space: parameters for the neural network weights
#        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
#
#        # Observation space: state variables (x, v, t)
#        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
#
#        # Constants
#        self.m = 1.0
#        self.Omega = 1.0
#        self.A = 1.0
#        self.sigma = 1.0
#        self.T = 10.0
#        self.dt = 0.01
#        self.x_star = 0.5
#        self.v_th = 0.01
#
#        self.t_values = np.linspace(1e-8, self.T, int(self.T/self.dt))
#        self.reset()
#
#        # Initialize neural network
#        self.xi_nn = XiNN()
#        self.optimizer = optim.Adam(self.xi_nn.parameters(), lr=0.001)
#
#    def reset(self):
#        self.x = 0.0
#        self.v = 0.0
#        self.current_step = 0
#        return np.array([self.x, self.v, self.current_step])
#
#    def force(self, x, xi):
#        return -(self.Omega**2 * (x - xi)) - (2.0 * self.A * (x - xi) / self.sigma) * np.exp(-(x - xi)**2 / self.sigma)
#
#    def rk4_step(self, x, v, xi_t, dt):
#        def dxdt(v):
#            return v
#        def dvdt(x, xi):
#            return self.force(x, xi) / self.m
#
#        k1_x = dxdt(v)
#        k1_v = dvdt(x, xi_t)
#
#        k2_x = dxdt(v + 0.5 * dt * k1_v)
#        k2_v = dvdt(x + 0.5 * dt * k1_x, xi_t)
#
#        k3_x = dxdt(v + 0.5 * dt * k2_v)
#        k3_v = dvdt(x + 0.5 * dt * k2_x, xi_t)
#
#        k4_x = dxdt(v + dt * k3_v)
#        k4_v = dvdt(x + dt * k3_x, xi_t)
#
#        x_new = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
#        v_new = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
#
#        return x_new, v_new
#
#    def xi(self, t):
#        t_tensor = torch.tensor([[t]], dtype=torch.float32)  # Convert to tensor
#        return self.xi_nn(t_tensor).item()  # Get scalar output
#
#    def step(self, action):
#        print("action =",action)
#        # Here, we can modify weights slightly as a form of action
##        with torch.no_grad():
##            # Update network weights using action
##            for param in self.xi_nn.parameters():
##                param += action[0] * param  # Update based on action
#
#        MSE = 0.0
#        x, v = self.x, self.v
#        for t in self.t_values:
#            xi_t = self.xi(t)
#            x, v = self.rk4_step(x, v, xi_t, self.dt)
#
#        # Calculate loss
#        MSE = (self.x_star - x)**2
#        abs_v = abs(v)
#        if abs_v > self.v_th:
#            MSE += (abs_v - self.v_th)**2
#        diff_xi_T = self.x_star - self.xi(self.T)
#        MSE += diff_xi_T**2
#        diff_xi_0 = self.xi(0)
#        MSE += diff_xi_0**2
#
#        # Reward is negative loss (minimize loss)
#        reward = -MSE
#
#        # Update state
#        self.x, self.v = x, v
#        self.current_step += 1
#
#        done = self.current_step >= len(self.t_values)
#        info = {}
#
#        return np.array([self.x, self.v, self.current_step]), reward, done, info
#
#'''
#build_actor
#===========
#Builds Actor deep-neural network for DDPG agent using the
#Taken from the following source: https://github.com/keras-rl/keras-rl/blob/master/examples/ddpg_pendulum.py
#'''
#
#
#def build_actor(env):
#    nb_actions = env.action_space.shape[0]
#    actor = Sequential()
#    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
#    actor.add(Dense(16))
#    actor.add(Activation('relu'))
#    actor.add(Dense(16))
#    actor.add(Activation('relu'))
#    actor.add(Dense(16))
#    actor.add(Activation('relu'))
#    actor.add(Dense(nb_actions))
#    actor.add(Activation('linear'))
#    return actor
#
#
#'''
#build_critic
#============
#Builds critic deep-neural network for DDPG agent
#Taken from the following source: https://github.com/keras-rl/keras-rl/blob/master/examples/ddpg_pendulum.py
#'''
#
#
#def build_critic(env):
#    nb_actions = env.action_space.shape[0]
#    action_input = Input(shape=(nb_actions,), name='action_input')
#    observation_input = Input(
#        shape=(1,) + env.observation_space.shape, name='observation_input')
#    flattened_observation = Flatten()(observation_input)
#    x = Concatenate()([action_input, flattened_observation])
#    x = Dense(16)(x)
#    x = Activation('relu')(x)
#    x = Dense(1)(x)
#    x = Activation('linear')(x)
#    critic = Model(inputs=[action_input, observation_input], outputs=x)
#    return critic, action_input
#
#'''
#build_agent
#============
#Builds DDPG agent using actor and critic deep-neural networks
#Taken from the following source: https://github.com/keras-rl/keras-rl/blob/master/examples/ddpg_pendulum.py
#'''
#
#
#def build_agent(env, actor, critic, action_input):
#    nb_actions = env.action_space.shape[0]
#    memory = SequentialMemory(limit=int(1e5), window_length=1)
#    ddpg = DDPGAgent(nb_actions, actor, critic, action_input,
#                     memory=memory, batch_size=32)
#    return ddpg
#
#if __name__ == "__main__":
#
#    test = CustomEnv()
#
#    actor = build_actor(test)  # actor neural network
#    actor.summary()  # summary of actor neural network architecture
#    critic, action_input = build_critic(test)  # critic neural network
#    critic.summary()  # summary of critic neural network architecture
#
#    ddpg = build_agent(test, actor, critic, action_input)
#    # Giving actor and critic neural networks Adam optimizers with learning
#    # rate 1e-5 and 1e-4 respectively. Generally a good idea to make the
#    # actor a slower learner than the critic. See the brief explanation
#    # here:
#    # https://www.reddit.com/r/reinforcementlearning/comments/lsza9m/why_different_learning_rates_for_actor_and_critic/
#    ddpg.compile([LazyAdam(1e-5), LazyAdam(1e-4)])
#    ddpg.fit(test, nb_steps=1e10, visualize=False)#,callbacks=[adjust_model])
#

