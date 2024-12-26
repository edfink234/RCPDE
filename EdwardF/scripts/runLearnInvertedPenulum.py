from os import system
import numpy as np
for x_0 in [1.5, 1.0, -0.5]:#np.linspace(-1.0, 1.5, 6):
    with open("temp.txt", "w") as f:
        f.write(f"{x_0}\n")
    system("python LearnInvertedPenulum.py")
