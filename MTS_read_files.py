#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 05:41:22 2022

A script to load MTS like data

@author: vinicius
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def interp_sampling(df, n):
    # Suggestion: describe the input variables
    # MTS
    if df['Load'].mean() < 0:
        df['Load'] = -df['Load']

    aux1 = np.linspace(0, df['Load'].idxmax(), n//2, dtype=int)
    aux2 = np.linspace(df['Load'].idxmax(), df['Load'].size,
                       (n//2)+1, dtype=int)
    aux = np.unique(np.concatenate((aux1, aux2)))
    sample = pd.DataFrame()
    for name in df.columns:
        sample[name] = np.interp(aux, df.iloc[:, 0], df[name])

    return sample


# Sensor and piston areas
piston_area = 30*30*0.25*np.pi

# load calibration Lynx data
MTS_files = ['Sample.txt']

dMTS = pd.DataFrame()

f = 0
sampling_n = 10000
disp_rate = ['04']

for file in MTS_files:

    data = pd.read_csv(MTS_files[f], sep='\t', header=5,
                       names=(["Time", "Crosshead", "Load", "Camera"]),
                       index_col=False, decimal=',')
    data = interp_sampling(data, sampling_n)

    dMTS[f"Time_{disp_rate[f]}"] = data["Time"]/3600  # hours
    dMTS[f"Crosshead_{disp_rate[f]}"] = data["Crosshead"]
    dMTS[f"Load_{disp_rate[f]}"] = (data["Load"]*1000/piston_area)
    dMTS[f"Load_percent_{disp_rate[f]}"] = (dMTS[f"Load_{disp_rate[f]}"] /
                                            dMTS[f"Load_{disp_rate[f]}"].max())
    dMTS[f"Load_percent_{disp_rate[f]}"] = dMTS[
        f"Load_percent_{disp_rate[f]}"] * 100
    dMTS[f"Test_rate{disp_rate[f]}"] = disp_rate[f]

    f += 1

dMTS.to_csv('MTS_posprocessado.csv', sep='\t')

# plot MTS results Load vs Time
plt.figure()
plt.plot(dMTS[f"Time_{disp_rate[0]}"], dMTS[f"Load_{disp_rate[0]}"],
         label=f"MTS {disp_rate[0]}")
plt.ylabel("Nominal Stress [MPa]")
plt.xlabel("Time [h]")
plt.legend()

# plot MTS results displacement vs Time

plt.figure()
plt.plot(dMTS[f"Time_{disp_rate[0]}"], dMTS[f"Crosshead_{disp_rate[0]}"],
         label=f"MTS {disp_rate[0]}")
plt.ylabel("Crosshead [mm]")
plt.xlabel("Time [h]")
plt.legend()

