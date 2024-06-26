import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

DATASET = "dexter"
PRECISION = "16"
PATH = str(Path.cwd()) + "/results/"

fileNames = sorted(os.listdir(PATH))
fileNames = [file for file in fileNames if DATASET and ("_" + PRECISION + "_") in file]
usecols = ["emissions"]
x = np.arange(len(usecols))
width = 0.15
multiplier = 0
keys = []
fig, ax = plt.subplots(layout='constrained')

for file in fileNames:
    fileId = "_" + file.split("_")[1].split(".")[0]
    df = pd.read_csv(PATH + file, usecols=usecols)
    keys = df.keys()

    df["emissions"] = df["emissions"] * 1000    
    
    offset = width * multiplier
    rects = ax.bar(x + offset, df.values[0], width, label=file.split("_")[3].split(".")[0] + " features")
    ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set_ylabel("CO2 Kg")
ax.set_title(DATASET + " float" + PRECISION + " emission comparison")
ax.tick_params(labelbottom=False)
ax.legend(loc='upper left')
ax.set_ylim(0, 250)

plt.show()
