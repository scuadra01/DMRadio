# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 08:54:38 2024

@author: sergio
"""
import h5py
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, sosfiltfilt, butter
from scipy.optimize import curve_fit
import gc
from memory_profiler import profile

N_SAMPLES = 256 * 1024 * 1024  # when cold
N_CHAN = 2  # 3 is not allowed!
SAMPLE_RATE = 10000000 #DONT TOUCH
DURATION = int(N_SAMPLES) / (N_CHAN * SAMPLE_RATE)
TIME = np.linspace(0, DURATION, num=int(N_SAMPLES))  
Nq = 0  # variable for the total  number of
Q_final = []  # list for the final Q
F_final = []
Ni = 0

makePlot = False

#@profile
def read_keys(filepath):

    data = h5py.File(str(filepath), 'r')

    print("Root groups:", list(data.keys()))

    dataDict = {}


    def readKeys(group):
        for key, value in group.items():
            if isinstance(value, h5py.Dataset):
                sys.stdout.write((value.name) + " --> dataset found, added to dictionary\n")
                dataDict[value.name] = value[()].flatten()


    full_key = ""
    readKeys(data)
    data.close()

    
    return dataDict 
#@profile
def read_keys(filepath):
    data_dict = {}
    with h5py.File(filepath, 'r') as data:
        for key, value in data.items():
            if isinstance(value, h5py.Dataset):
                print(f"{value.name} --> dataset found, adding to dictionary")
                # Consider whether flattening is necessary based on your use case
                data_dict[value.name] = value[()]  # Don't flatten if not necessary
    return data_dict


#@profile
def initialize_data(dataDict, N_CHAN):

    dataTime = {}

    dataTime = np.array([TIME[chan:N_SAMPLES:N_CHAN] for chan in range(N_CHAN)])
    timeBurst, timeSync = dataTime

    voltageBurst = dataDict["/voltageBurst"].astype(np.float32)
    voltageSync = dataDict["/voltageSync"].astype(np.float32)

    
    if makePlot:
        plt.plot(timeBurst, voltageBurst)
        # plt.plot(timeSync,voltageSync)
        plt.xlabel("Times [s]")
        plt.ylabel("Voltage [V]")
        plt.grid()
        plt.savefig("raw_data_NEW.png")
        plt.show(block=False)
    return voltageBurst, voltageSync, timeBurst, timeSync 

#@profile
def hilbertTransform(input):  # make Hilbert transformation, to get only the amplitude envelope of the signal
    analytic_signal = hilbert(input)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope
#@profile   
def Rise_Fall(vB, vS, tB, tS):
    '''
    vB = voltageBurst
    vS - voltageSync
    tB = timeBurst
    tS = timeSync
    '''  


    trigger_val = 3.25

    # Find indices where voltage transitions from above to below trigger_val (falling edges)
    falling_edges_mask = (vS[:-1] > trigger_val) & (vS[1:] < trigger_val)
    falling_edges_indices = np.flatnonzero(falling_edges_mask) + 1

    # Find indices where voltage transitions from below to above trigger_val (rising edges)
    rising_edges_mask = (vS[:-1] < trigger_val) & (vS[1:] > trigger_val)
    rising_edges_indices = np.flatnonzero(rising_edges_mask) + 1

    # Ensure rising and falling edges are aligned properly
    if rising_edges_indices[0] < falling_edges_indices[0]:
        rising_edges_indices = rising_edges_indices[1:]

    if falling_edges_indices[-1] > rising_edges_indices[-1]:
        falling_edges_indices = falling_edges_indices[:-1]

    # Pair falling and rising edges
    ringdowns = list(zip(falling_edges_indices, rising_edges_indices))
    return ringdowns

#@profile    
def extract_ringdown(start, finish, tB, vB, vB_H):
    ni = Ni 
    ni += 1

    ringdownTime = tB[start:finish]
    ringdownData = vB_H[start:finish]

    # we take two e-foldings starting from the highest voltage
    ringdownDataArg = np.argwhere(ringdownData > (ringdownData[0] / np.e ** 2)).T.flatten()

    # throw away data that lies far away from the main portion
    ringdownDataArgGroups = np.split(ringdownDataArg, np.where(np.diff(ringdownDataArg) != 1)[0] + 1)

    ringdownDataArgGroups.sort(key=len)  # keep only the the greatest portion of data
    while len(ringdownDataArgGroups) > 1:
        ringdownDataArgGroups.pop(0)

    ringdownDataArg = ringdownDataArgGroups[0]

    ringdownTime = tB[start + ringdownDataArg]  # time array for this ringdown
    ringdownData = vB[start + ringdownDataArg]  # voltage for this ringdwon



    if makePlot:
        plt.plot(ringdownTime, ringdownData)
        plt.xlabel("Times [s]")
        plt.ylabel("Voltage [V]")
        plt.show(block=False)
        plt.clf()
    
    return ringdownTime, ringdownData
#@profile    
def psd_ringdown(ringdownTime, ringdownData):
    
    N_SAMPLES = np.shape(ringdownTime)[0]  # number of samples for this ringdown
    TIME_STEP = round(ringdownTime[1] - ringdownTime[0], 10)  # time step for this ringdown
    SAMPLE_RATE = 1 / TIME_STEP
    DURATION = N_SAMPLES / SAMPLE_RATE
    FREQ_STEP = 1 / DURATION

    fft = np.fft.rfft(ringdownData)
    fftfreq = np.fft.rfftfreq(N_SAMPLES, TIME_STEP)
    psd = (np.square(np.abs(fft))) / (N_SAMPLES ** 2 * FREQ_STEP)  # find PSD
    peak_x = np.argmax(psd)


    if makePlot:
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD [V/srHz]")
        plt.grid()
        plt.loglog(fftfreq, psd)
        # plt.savefig(savepath + "psd_{:.0f}.pdf".format(ni))
        plt.show(block=False)
        plt.clf()
    return fftfreq, peak_x, SAMPLE_RATE

#@profile
def Q_Calc(ringdownTime, ringdownData, fftfreq, peak_x, SAMPLE_RATE):
    Q_start_time = time.time()
    F_PEAK = fftfreq[peak_x]
    tau_estimate = (ringdownTime[-1] - ringdownTime[0]) / 2  # estimate for the relaxation time
    print("Resonance frequency = {:,.1f} Hz".format(F_PEAK))

    Q_estimate = np.pi * F_PEAK * tau_estimate
    print("Q estimate = {:,.0f}".format(Q_estimate))

    if (300050 < F_PEAK < 300110) and (2000 < Q_estimate < 400000):
        nq = Nq
        nq += 1
        dF = F_PEAK / Q_estimate

        Q1, Q2 = 3500, 250000
        df1, df2 = 0.05, 0.005

        # depending on the value of Q, different df should be chosen, where we use df for filter
        k = (df1 - df2) / (Q1 - Q2)
        b = - (Q2 * df1 - Q1 * df2) / (Q1 - Q2)

        if Q_estimate < Q1:
            df = 0.05
        elif Q_estimate > Q2:
            df = 0.004
        else:
            df = k * Q_estimate + b

        # Precompute filter parameters
        lowF = F_PEAK * (1 - df)
        highF = F_PEAK * (1 + df)
        sos = butter(100, [lowF, highF], btype='band', fs=SAMPLE_RATE, output='sos')

        # Filter the signal
        filtered = sosfiltfilt(sos, ringdownData)
        filter_time = time.time()
        print("filter_time: {:.3f} ms".format((filter_time - Q_start_time) * 1e3))

        # Perform Hilbert transformation of the filtered signal
        analytic_signal = hilbert(filtered)
        amplitude_envelope = np.abs(analytic_signal)  # amplitude envelope of the filtered signal

        amplitude_envelope_time = time.time()
        print("amplitude_envelope_time: {:.3f} ms".format((amplitude_envelope_time - filter_time) * 1e3))

        N = len(ringdownTime)
        cut_ratio = 0.15
        cut_size = int(cut_ratio * N)
        ringdownTime = ringdownTime[cut_size:-cut_size]
        amplitude_envelope = amplitude_envelope[cut_size:-cut_size]



        # Initial Guess Optimization
        aguess = amplitude_envelope[0]
        bguess = tau_estimate
        guess = [aguess, bguess]

        # Fitting Function
        #@profile
        def func(x, a, b):
            return a * np.exp(-x / b)

        #exp fit
        popt, pcov = curve_fit(func, ringdownTime - ringdownTime[0], amplitude_envelope, guess)
        a, b = popt


        db = np.sqrt(pcov[1][1])
        Q = np.pi * F_PEAK * b
        dQ = np.pi * F_PEAK * db

        print("Q from exponential fit = {:,.0f}".format(Q))

        exponential_fit_time = time.time()
        print("exponential_fit_time: {:.3f} ms".format((exponential_fit_time - amplitude_envelope_time) * 1e3))

        if makePlot:
            plt.plot(ringdownTime - ringdownTime[0], ringdownData, label="experimental data")
            plt.plot(ringdownTime - ringdownTime[0], filtered, label="filtered experimental data")
            plt.plot(ringdownTime - ringdownTime[0], amplitude_envelope, label="amplitude envelope")
            plt.plot(ringdownTime - ringdownTime[0], func(ringdownTime - ringdownTime[0], *popt), label="fit")
            plt.xlabel("Time [s]")
            plt.ylabel("Voltage [V]")
            plt.grid()
            plt.legend(loc='upper right')
            plt.show(block=False)
            # plt.savefig(savepath + "ringdown_exp_{:.0f}.pdf".format(ni))
            plt.clf()

            plt.plot(ringdownTime - ringdownTime[0], amplitude_envelope, label="amplitude envelope")
            plt.plot(ringdownTime - ringdownTime[0], func(ringdownTime - ringdownTime[0], *popt), label="fit")
            plt.xlabel("Time [s]")
            plt.ylabel("Voltage [V]")
            plt.grid()
            plt.legend(loc='upper right')
            plt.show(block=False)
            # plt.savefig(savepath + "ringdown_exp2_{:.0f}.pdf".format(ni))
            plt.clf()

        # linear fit
        p, cov = np.polyfit(ringdownTime, np.log(amplitude_envelope), 1, cov=True)

        linear_fit_time = time.time()
        print("linear_fit_time: {:.3f} ms".format((linear_fit_time - exponential_fit_time) * 1e3))

        if makePlot:
            plt.plot(ringdownTime, np.log(amplitude_envelope))
            plt.plot(ringdownTime, np.polyval(p, ringdownTime))
            plt.xlabel("Time [s]")
            plt.ylabel("log of Voltage [V]")
            plt.grid()
            plt.show(block=False)
            # plt.savefig(savepath + "ringdown_lin_{:.0f}.pdf".format(ni))
            plt.clf()

        tau = -1 / p[0]

        # linear fit is more reliable, we use it for the final Q
        Q = np.pi * F_PEAK * tau
        global Q_final
        global F_final
        Q_final.append(Q)
        F_final.append(F_PEAK)
        print("Q from linear fit = {:,.0f}".format(Q))
        print("Resonance frequency = {:,.0f}".format(F_PEAK))
        print("================================")
    return nq

#@profile
def ringdownAnalysis():

    # folder = input("Enter folder name:\n")

    
    filepath ="output.h5"
    
    Timings = [[],[]]
    start_time = time.time()

    dataDict = read_keys(filepath)
    
    read_keys_time = time.time()
    print("read_keys_time: {:.3f} ms".format((read_keys_time - start_time) * 1e3))
    Timings[0].append("read_keys_time")
    Timings[1].append("{:.3f} ms".format((read_keys_time - start_time) * 1e3))
    print("processing...")
    
        
    vB, vS, tB, tS = initialize_data(dataDict, N_CHAN)
    time_voltage_gen_time = time.time()
    print("time_volt_gen_time: {:.3f} ms".format((time_voltage_gen_time - read_keys_time) * 1e3))
    Timings[0].append("time_volt_gen_time")
    Timings[1].append("{:.3f} ms".format((time_voltage_gen_time - read_keys_time) * 1e3))
    
    vB_H = hilbertTransform(vB)
    hilbert_transform_time = time.time()
    print("hilbert_transform_time: {:.3f} ms".format((hilbert_transform_time - time_voltage_gen_time) * 1e3))
    Timings[0].append("hilbert_transform_time")
    Timings[1].append("{:.3f} ms".format((hilbert_transform_time - time_voltage_gen_time) * 1e3))
    
    ringdowns = Rise_Fall(vB, vS, tB, tS)
    falling_rising_edges_time = time.time()
    print("falling_rising_edges_time: {:.3f} ms".format((falling_rising_edges_time - hilbert_transform_time) * 1e3))
    Timings[0].append("falling_rising_edges_time")
    Timings[1].append("{:.3f} ms".format((falling_rising_edges_time - hilbert_transform_time) * 1e3))
    start_ringdown_processing = time.time()
    for start, finish in ringdowns:
        start_for_cycle_time = time.time()
        ringdownTime, ringdownData = extract_ringdown(start, finish, tB, vB, vB_H)
        extract_one_ringdown_time = time.time()
        print("extract_one_ringdown_time: {:.3f} ms".format((extract_one_ringdown_time - start_for_cycle_time) * 1e3))
        Timings[0].append("extract_one_ringdown_time")
        Timings[1].append(" {:.3f} ms".format((extract_one_ringdown_time - start_for_cycle_time) * 1e3))

        fftfreq, peak_x, SAMPLE_RATE = psd_ringdown(ringdownTime, ringdownData)
        psd_time = time.time()
        print("psd_time: {:.3f} ms".format((psd_time - extract_one_ringdown_time) * 1e3))
        Timings[0].append("psd_time")
        Timings[1].append(" {:.3f} ms".format((psd_time - extract_one_ringdown_time) * 1e3))
        
        Nq = Q_Calc(ringdownTime, ringdownData, fftfreq, peak_x, SAMPLE_RATE)

    if Nq != 0:
        global Q_final
        global F_final
        Q_final = np.array(Q_final)
        Q_final_mean = Q_final.mean()
        Q_final_std = np.std(Q_final)
        print("Final Q = {:,.3f} +- {:,.3f}".format(Q_final_mean, Q_final_std))

        F_final = np.array(F_final)
        F_final_mean = F_final.mean()
        F_final_std = np.std(F_final)
        print("Final Q = {:,.0f} +- {:,.0f}".format(F_final_mean, F_final_std))
        ringdown_processing_time = time.time()
        print("ringdown_processing_time: {:.3f} ms".format((ringdown_processing_time - start_ringdown_processing) * 1e3))
    else:
        print("Bad data")

    sys.stdout.write('--- {:.0f} seconds ---\n'.format((time.time() - start_time)))
    





ringdownAnalysis()
gc.collect()
for name in dir():
    if not name.startswith('_'):
        del globals()[name]