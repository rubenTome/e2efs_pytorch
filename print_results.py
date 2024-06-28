import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def printCSV(dataset, precision, directory="results/"):

    PATH = str(Path.cwd()) + "/" + directory
    fileNames = sorted(os.listdir(PATH))
    fileNames = [file for file in fileNames if dataset and ("_" + precision + "_") in file]
    multiplier = 0
    keys = []
    fig, ax = plt.subplots(layout='constrained')

    for file in fileNames:
        fileId = "_" + file.split("_")[1].split(".")[0]
        df = pd.read_csv(PATH + file, usecols=["emissions"])
        keys = df.keys()
        df["emissions"] = df["emissions"] * 1000    
        offset = 0.15 * multiplier
        rects = ax.bar([0] + offset, df.values[0], 0.15, label=file.split("_")[3].split(".")[0] + " features")
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel("CO2 (g)")
    ax.set_title(dataset + " float" + precision + " emission comparison")
    ax.tick_params(labelbottom=False)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 250)

    plt.savefig()
    plt.clf()
    plt.cla()
