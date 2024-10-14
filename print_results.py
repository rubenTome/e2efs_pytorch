import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def printCSV(dataset, precision, directory="results/"):

    PATH = str(Path.cwd()) + "/" + directory
    fileNames = sorted(os.listdir(PATH))
    fileNames = [file for file in fileNames 
                 if "emissions_" + dataset + "_" in file 
                 and "_" + precision + "_" in file]
    multiplier = 0
    _, ax = plt.subplots(layout='constrained')

    for file in fileNames:
        df = pd.read_csv(PATH + file, usecols=["emissions"])
        df["emissions"] = df["emissions"] * 1000    
        offset = 0.15 * multiplier
        rects = ax.bar(offset, df.values[0], 0.15, label=file.split("_")[3].split(".")[0] + " features")
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel("CO2 (g)")
    ax.set_title(dataset + " float" + precision + " emission comparison")
    ax.tick_params(labelbottom=False)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 250)

    plt.savefig("plots/" + dataset + "_" + precision + "_nfeatures.png")
    plt.clf()
    plt.cla()
    plt.close()

datasets = ["dexter", "gina", "gisette", "madelon", "colon", "leukemia", "lung181", "lymphoma"]
precisions = ["16-mixed", "32", "64"]
for dataset in datasets:
    for precision in precisions:
        printCSV(dataset, precision)