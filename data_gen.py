import math
from stat import filemode
import pandas as pd
import numpy as np
import os
import tqdm
import glob
import re

p_f = re.compile('\d+\.\d+')
p_d = re.compile('\d+')
p_txPower = re.compile('^\d+\.\d+txPower')
p_distance = re.compile('txPower-\d+\.\d+m')
p_bw = re.compile('m\d+BW')
p_subbw = re.compile('BW\d+sub_BW')
p_suboff = re.compile('sub_BW\d+sub_off')
p_numIntfBS = re.compile('\d+_intfBS*')
p_x = re.compile('paras_\d+\.csv')

def match(filename):
    res = {}
    # res['txPower'] = float(re.search(p_f, p_txPower.search(filename).group()).group())
    # res['distance'] = float(re.search(p_f, p_distance.search(filename).group()).group())
    # res['bw'] = int(re.search(p_d, p_bw.search(filename).group()).group())
    # res['subbw'] = int(re.search(p_d, p_subbw.search(filename).group()).group())
    # res['suboff'] = int(re.search(p_d, p_suboff.search(filename).group()).group())
    # res['num_intfBS'] = int(re.search())
    res['num_intfBS'] = (int(p_d.search(p_numIntfBS.search(filename).group()).group()))
    return res

def gen(folder, minmax):
    num_intfBS = match(folder)['num_intfBS'] 
    if folder[-1] != '/':
        folder = folder + '/'
    files = os.listdir(folder)
    idxes = set()
    for f in files:
        if p_x.search(f) != None:
            idxes.add(int(p_d.search(f).group()))
    N = len(idxes)
    x = np.empty((N, num_intfBS, 5))
    yhat = np.empty((N, 1))
    j = 0
    for i in tqdm.tqdm(idxes):
        px = folder + "paras_" + str(i) + ".csv"
        dfx = pd.read_csv(px)
        xi = dfx[dfx.columns[0:5]].to_numpy()
        x[j] = xi
        py = folder + str(i) + "-DL-SINR.csv"
        dfy = pd.read_csv(py)
        sinr_max = max(dfy["SINR"].unique())
        sinr_min = min(dfy["SINR"].unique())
        sinr_drop = sinr_max - sinr_min
        yhat[j] = sinr_drop / sinr_max
        j = j + 1
    np.save(folder + 'x.npy', x)
    np.save(folder + 'y.npy', yhat)

folder = ""

if __name__ == '__main__':
    for d in os.listdir(folder):
        print(folder + d)
        gen(folder + d, None)
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for f in filenames:
            if f.endswith('.npy'):
                os.system('ls -ahl ' + dirpath + '/' + f)