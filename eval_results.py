"""
    the script is to calucate the average accuracy of evaluation results instead
    of munally selected the one
"""

import os
import sys
import fire
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def extract_detection_accuracy(results_dir, gen_start, gen_end):
    # reading all the data
    all_data = np.loadtxt(results_dir, dtype=bytes, delimiter="\n").astype(str)
    num_data = len(all_data)
    ap_3d = []
    gens = []
    max_easy = 0
    max_midd = 0
    max_hard = 0

    for i in range(num_data):
        if all_data[i][:10]=='Generation':
            gen = int(all_data[i][11:])
            _ind_2 = 0
            # fetching desired detection results
            if gen > gen_start and gen < gen_end:
                _temp_str = all_data[i+1]
                _ind_1 = np.char.find( _temp_str,'([')
                _ind_1 += 2
                _temp_str = _temp_str[_ind_1:]
                _ind_2 = np.char.find(_temp_str,'])')
                easy_ap = float(_temp_str[:_ind_2])
                if easy_ap>max_easy:
                    max_easy = easy_ap

                _temp_str = _temp_str[_ind_2:]
                _ind_1 = np.char.find( _temp_str,'([')
                _ind_1 += 2
                _temp_str = _temp_str[_ind_1:]
                _ind_2 = np.char.find(_temp_str,'])')
                midd_ap = float(_temp_str[:_ind_2])

                if midd_ap > max_midd:
                    max_midd = midd_ap

                _temp_str = _temp_str[_ind_2:]
                _ind_1 = np.char.find( _temp_str,'([')
                _ind_1 += 2
                _temp_str = _temp_str[_ind_1:]
                _ind_2 = np.char.find(_temp_str,'])')
                hard_ap = float(_temp_str[:_ind_2])

                avg_rec = float(all_data[i+2][25:])

                if hard_ap > max_hard:
                    max_hard = hard_ap
                ap_3d.append([easy_ap, midd_ap, hard_ap, avg_rec])
                gens.append(gen)

    # calculating average detection accuracy
    ap_3d = np.asarray(ap_3d).reshape((-1,4))
    gens = np.asarray(gens).reshape((-1,1))
    return ap_3d, [max_easy, max_midd, max_hard], gens


def avg_eval(results_dir,
             start_steps = 60e3,
             end_steps = 200e3,
            ):
    # checking the result path
    if not pathlib.Path(results_dir).exists():
        raise ValueError("The result does not exist!")

    # reading all the data
    #all_data = np.loadtxt(results_dir, dtype=bytes, delimiter="\n").astype(str)
    #num_data = len(all_data)
    #ap_3d = []
    #for i in range(num_data):
    #    if all_data[i][:8]=='Eval_at_':
    #        steps = int(all_data[i][8:])
    #        # fetching desired detection results
    #        if steps > start_steps and steps < end_steps:
    #            easy_ap = float(all_data[i+4][8:13])
    #            midd_ap = float(all_data[i+4][15:20])
    #            hard_ap = float(all_data[i+4][22:27])
    #            ap_3d.append([easy_ap, midd_ap, hard_ap])

    ## calculating average detection accuracy
    #ap_3d = np.asarray(ap_3d).reshape((-1,3))
    ap_3d, max_ap,  _ = extract_detection_accuracy(results_dir, start_steps, end_steps)
    avg_ap_3d = np.around(np.mean(ap_3d, axis = 0), decimals=2)

    print("the average detection AP of 3D objects:")
    print(avg_ap_3d)
    print("\nthe max detection AP of 3D object:")
    print(max_ap)


def plot_accuracy_curve(results_dir,
                        gen_start = 0,
                        gen_end = 800,):

    # checking the result path
    if not pathlib.Path(results_dir).exists():
        raise ValueError("The result does not exist!")

    recall, _, steps = extract_detection_accuracy(results_dir, gen_start, gen_end)


    fig, ax = plt.subplots()
    easy_ap = ax.plot(steps, recall[:,0], label='easy recall')
    midd_ap = ax.plot(steps, recall[:,1], label='midd recall')
    hard_ap = ax.plot(steps, recall[:,2], label='hard recall')
    hard_ap = ax.plot(steps, recall[:,3], label='averge recall')
    ax.legend()
    plt.savefig(results_dir.replace(".txt",".png"))
    # plotting the detection accuracy


def plot_pos_figure(results_dir):
    # read data
    df = pd.read_csv(results_dir)
    data = [list(x[[0,1,2,3,5,6,7]]) for x in df.values]
    data = np.array(data)
    num = data.shape[0]
    x = np.arange(0,2*num,2)

    fig, ax = plt.subplots()
    easy_ap = ax.plot(x,100*data[:,2], label='Easy')
    midd_ap = ax.plot(x,100*data[:,1], label='Midd')
    hard_ap = ax.plot(x,100*data[:,0], label='Hard')
    hard_ap = ax.plot(x,100*data[:,3], label='Averge')
    ax.legend(loc=4)
    ax.set_ylim(75, 100)
    plt.savefig(results_dir.replace(".csv","recall.png"))


    fig, ax = plt.subplots()
    easy_ap = ax.plot(x, data[:,4], label='D_offset')
    midd_ap = ax.plot(x, data[:,5], label='H_dist')
    hard_ap = ax.plot(x, data[:,6], label='V_dist')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.savefig(results_dir.replace(".csv","paramters.png"))







if __name__=='__main__':
    fire.Fire()



