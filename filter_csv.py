from scipy import signal
import sys
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import csv
def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def readCSVToNumpyArray(dataset):
        values = [[]]
        with open(dataset) as f:
            counter = 0
            for i in csv.reader(f):
                for j in i:
                    try:
                        values[counter].append(float(j))
                    except ValueError:
                        values[counter].append(j)
                counter = counter + 1
                values.append([])

        data = np.array(values[:-1],dtype='object')

        return data


inp=sys.argv[1]
filename = "l2cs_"+osp.splitext(osp.basename(inp))[0]
output=sys.argv[2]
cutoff=float(sys.argv[3])
nyq=float(sys.argv[4])
order=int(sys.argv[5])
#filename = "l2cs_"+osp.splitext(osp.basename(inp))[0]
filename_full = "filtered_"+osp.basename(osp.join(output, filename + "_gaze.csv"))
data = readCSVToNumpyArray(osp.join(output, filename + "_gaze.csv"))
print(osp.join(output, filename + "_gaze.csv"))
data[1:,1]=butter_lowpass_filter(data[1:,1],cutoff, nyq,order)
#print(data['pitch'])
data[1:,2]=butter_lowpass_filter(data[1:,2], cutoff, nyq,order)
import pandas as pd 
pd.DataFrame(data).to_csv(osp.join(output,filename_full),header=None,index=False)
#np.savetxt(osp.join(output,filename_full), data, delimiter=",")