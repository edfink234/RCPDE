from os import system
import numpy as np
for x_0 in np.linspace(-1.5, 1.5, 7):
    with open("temp.txt", "w") as f:
        f.write(f"{x_0}\n")
    system("python LearnInvertedPenulum.py")
