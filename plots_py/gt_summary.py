import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from result_csv import *
# from img_result_csv import *

# import matplotlib.pyplot.hist as pyhist
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)


def plot_results(results_root ,str_name, var_list=["01"]):
        # plot
    fig, ax = plt.subplots()
    for var in var_list:
        all_results = []
        alL_dates = []
        results_dict = globals()["{}".format(var)]
        for key, value in results_dict.items():
            all_results.append(value)
            alL_dates.append(x_days[key])
        x = list(np.arange(len(alL_dates)))
        if "pc" in var:
            plt.plot(x, all_results,  color=color_maps[var], linestyle='-', linewidth=1, marker='o', markersize=10, label=key2names[var])
        else:
            plt.plot(x, all_results,  color=color_maps[var], linestyle='-.', linewidth=1, marker='*', markersize=10, label=key2names[var])
    dates = results_dict.keys()
    plt.xticks(x, dates)
    plt.xlabel("Date")
    plt.ylabel("MARE")
    plt.ylim([0.0, 0.6])
    plt.title("")
    ax.legend()
    plt.savefig(str(results_root) + "/pc_res.png")
    # plt.savefig(str(results_root) + "/img_res.png")



if __name__ == "__main__":
    results_root = "./figs"
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    plot_results(results_root, "our_method", var_list=["ours_pc", "bionet_pc", "dgcnn_pc", "ours_neff", "bionet_neff", "dgcnn_neff"])




