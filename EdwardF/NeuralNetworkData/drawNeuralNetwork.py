import torch
import torch.nn as nn
from torchviz import make_dot
from os import system
import sys

model_path = "xi_model_IC_2_point_2258_1_point_0_1_point_0_1_point_0_0_point_2_.pth"
# Define the neural network for xi(t)
class XiModel(nn.Module):
    def __init__(self):
        self.num_units = 4
        super(XiModel, self).__init__()
        # Initialize layers with reduced neurons
        self.fc1 = nn.Linear(1, self.num_units)
        self.fc2 = nn.Linear(self.num_units, self.num_units)
        self.fc3 = nn.Linear(self.num_units, self.num_units)
        self.fc4 = nn.Linear(self.num_units, self.num_units)
        self.fc_out = nn.Linear(self.num_units, 1)
        self.fc_skip = nn.Linear(1, 1)
        
    def forward(self, inputs):
        x = torch.tanh(self.fc1(inputs))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        output_main = self.fc_out(x)
        output_skip = self.fc_skip(inputs)
        return output_main + output_skip

# Instantiate the model
model = XiModel()
model.load_state_dict(torch.load(model_path, weights_only=True))
print(model)

example = torch.tensor([[0.5]], dtype=torch.float32)

# Forward pass
output = model(example)[0, 0]

# Generate visualization
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render("model_graph")

import subprocess

# Run dot2tex to convert the dot file to a TikZ file
subprocess.run(["dot2tex", "--format", "tikz", "model_graph", "-o", "model_graph.tex"])
