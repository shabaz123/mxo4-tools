# -*- coding: utf-8 -*-
"""
Audio Analyzer for MXO 4
main.py - rev 1 - shabaz
@author: shabaz
"""
import pyvisa
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import cmath
import numpy as np
import math
import time

# configuration parameters
DEVICE_ID = "192.168.1.87"

# test parameters
WGEN_VPP = 0.707  # Vpp of waveform generator output

# other constants
COUPLING_DC = 0
COUPLING_AC = 1
NUM_ACQ_POINTS = 1000
TWO_PI = 2.0 * cmath.pi
SQRT2 = math.sqrt(2.0)
SQRT50 = math.sqrt(50.0)
CMPLX1 = complex(1.0, 0.0)
ONE = 1.0
FREQVAL = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
           200, 300, 400, 500, 600, 700, 800, 900, 1000,
           2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
           20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

# globals
inst = None  # stores handle to instrument
inst_name = ""  # example is "TCPIP::192.168.1.66::INSTR"
c1data = []  # stores channel 1 data
c2data = []  # stores channel 2 data
rm = pyvisa.ResourceManager()
samps_per_sec = 400000  # acquisition samples per second (Hz)
cycles_per_sec = 1000  # waveform generator frequency (Hz)
rad_per_sec = 1
rad_per_samp = 1
out_s = []
out_phase = []
c1vectors = []
c2vectors = []
freq1 = []
freq2 = []
real1 = []
real2 = []
imag1 = []
imag2 = []
vpp1 = []
vpp2 = []
vpp = []
deg1 = []
deg2 = []
deg = []
real = []
imag = []
s11 = []
phoffset1 = 0.0


# functions
def open_instrument():
    global inst
    global inst_name
    rl = rm.list_resources()
    if len(rl) > 0:
        for i in range(0, len(rl)):
            if DEVICE_ID in rl[i]:
                inst_name = rl[i]
                print("Instrument found: ")
                print(f"    {inst_name}")
                inst = rm.open_resource(inst_name)
                print(f"    {inst.query('*IDN?')}")
                print("Instrument opened")
                return 1
    return 0


def close_instrument():
    global inst
    if inst is not None:
        inst.close()
        inst = None
        print("Instrument closed")
    else:
        print("Instrument already closed")


def wait_for_opc():
    if inst is not None:
        for i in range(0, 20):
            dummy = inst.query_ascii_values('*OPC?')
            if dummy[0] == 1:
                return 1
            time.sleep(0.1)
    print("Error, instructions not completed!")
    return 0


def set_chan_params(chan, vdiv, offset, coupling):
    if inst is not None:
        print(f"Setting channel {chan} parameters...", end='')
        if coupling == COUPLING_DC:
            inst.write(f"Channel{chan}:COUPling DC")
        else:
            inst.write(f"Channel{chan}:COUPling AC")
        inst.write(f"Channel{chan}:BANDwidth B20")
        inst.write(f"Channel{chan}:SCALe {vdiv}")
        inst.write(f"Channel{chan}:OFFSet {offset}")
        inst.write(f"Channel{chan}:POSition 0.0")
        inst.write(f"Channel{chan}:STATe ON")
        wait_for_opc()
        print("done")


def set_acquisition_params():
    if inst is not None:
        print("Setting acquisition parameters...", end='')
        inst.write("FORMat:DATA REAL,32")
        inst.write("ACQuire:TYPE SAMPle")
        inst.write("ACQuire:INTerpolate SINX")
        inst.write("ACQuire:POINts:MODE MANual")
        inst.write(f"ACQuire:POINts {NUM_ACQ_POINTS}")
        wait_for_opc()
        print("done")


def set_timebase_range(range_ms):
    global samps_per_sec
    global rad_per_samp
    if inst is not None:
        inst.write(f"TIMebase:RANGe {range_ms / 1000.0}")
        wait_for_opc()
        samps_per_sec = NUM_ACQ_POINTS / (range_ms / 1000.0)
        rad_per_samp = rad_per_sec / samps_per_sec


def set_wavegen_freq(freq):
    global cycles_per_sec
    global rad_per_sec
    global rad_per_samp
    if inst is not None:
        inst.write(f"WGENerator1:FREQuency {freq}")
        wait_for_opc()
        cycles_per_sec = freq
        rad_per_sec = TWO_PI * cycles_per_sec
        rad_per_samp = rad_per_sec / samps_per_sec


def set_wavegen_params():
    if inst is not None:
        print("Setting wavegen parameters...", end='')
        inst.write("WGENerator1:OUTPut:LOAD HIZ")
        inst.write("WGENerator1:FUNCtion:SELect SINusoid")
        inst.write(f"WGENerator1:VOLTage:VPP {WGEN_VPP}")
        inst.write("WGENerator1:VOLTage:OFFSet 0.0")
        set_wavegen_freq(1000)
        inst.write("WGENerator1:ENABle ON")
        wait_for_opc()
        print("done")


def set_wavegen_vpp(vpp):
    if inst is not None:
        inst.write(f"WGENerator1:VOLTage:VPP {vpp}")
        wait_for_opc()


def set_wavegen_state(state):
    if inst is not None:
        if state == 0:
            inst.write("WGENerator1:ENABle OFF")
        else:
            inst.write("WGENerator1:ENABle ON")
        wait_for_opc()


def acquire_data():
    global c1data
    global c2data
    if inst is not None:
        print("Acquiring data...", end='')
        inst.write("RUNSingle")
        wait_for_opc()
        c1data = inst.query_binary_values('Channel1:DATA:VALues?')
        c2data = inst.query_binary_values('Channel2:DATA:VALues?')
        print("done")


# helper function, find nearest value in array
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def goertzel(samples):
    """
    See https://stackoverflow.com/questions/13499852/scipy-fourier-transform-of-a-few-selected-frequencies
    """
    sample_rate = samps_per_sec
    freqs = [(cycles_per_sec - 2, cycles_per_sec + 2)]
    window_size = len(samples)
    f_step = sample_rate / float(window_size)
    f_step_normalized = 1.0 / window_size
    print(
        f"sample_rate={sample_rate}, window_size={window_size}, f_step={f_step}, f_step_normalized={f_step_normalized}")

    # Calculate all the DFT bins we have to compute to include frequencies
    # in `freqs`.
    bins = set()
    for f_range in freqs:
        f_start, f_end = f_range
        k_start = int(math.floor(f_start / f_step))
        k_end = int(math.ceil(f_end / f_step))

        if k_end > window_size - 1:
            raise ValueError('frequency out of range %s' % k_end)
        bins = bins.union(range(k_start, k_end))

    # For all the bins, calculate the DFT term
    n_range = range(0, window_size)
    freqs = []
    results = []
    for k in bins:

        # Bin frequency and coefficients for the computation
        f = k * f_step_normalized
        w_real = 2.0 * math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)

        # Doing the calculation on the whole sample
        d1, d2 = 0.0, 0.0
        for n in n_range:
            y = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y

        # Storing results `(real part, imag part, power)`
        results.append((
            0.5 * w_real * d1 - d2, w_imag * d1,
            d2 ** 2 + d1 ** 2 - w_real * d1 * d2)
        )
        freqs.append(f * sample_rate)
    # find the closest freq and return just that result
    return [freqs[find_nearest_idx(freqs, cycles_per_sec)]], [results[find_nearest_idx(freqs, cycles_per_sec)]]


def plot_channel(chan):
    plt.interactive(False)
    if chan == 1:
        plt.plot(c1data, label="Channel 1")
    elif chan == 2:
        plt.plot(c2data, label="Channel 2")
    plt.legend()
    plt.show()


def do_scan():
    global c1vectors
    global c2vectors
    global NUM_ACQ_POINTS
    c1vectors = []
    c2vectors = []
    NUM_ACQ_POINTS = 10000
    freq = 200000
    set_chan_params(1, 1.0, 0.0, COUPLING_AC)  # chan, vdiv, offset, impedance
    set_chan_params(2, 1.0, 0.0, COUPLING_AC)
    set_wavegen_params()
    set_wavegen_vpp(WGEN_VPP)
    totfreq = len(FREQVAL)
    imax = 50
    for i in range(0, totfreq):
        freq = FREQVAL[i]
        if freq < 1000:
            print(f"({i}/{totfreq - 1}): Acquiring at {freq} Hz")
        else:
            print(f"({i}/{totfreq - 1}): Acquiring at {freq / 1000} kHz")
        tb_range = (1 / freq) * 1000.0 * 10.0  # 10 periods
        set_timebase_range(tb_range)
        set_acquisition_params()
        set_wavegen_freq(freq)
        time.sleep(0.1)  # wait for wavegen output to be valid
        acquire_data()  # acquire chan 1 and chan 2 samples

        meas_freq = [0, 0]
        meas_vector = [0, 0]
        meas_vpp = [0, 0]
        meas_error = 0

        # channel 1:
        freqs, results = goertzel(c1data)
        if len(freqs) == 1:
            meas_freq[0] = freqs[0]
            meas_vector[0] = results[0]
        else:
            meas_error = 1
        # channel 2:
        freqs, results = goertzel(c2data)
        if len(freqs) == 1:
            meas_freq[1] = freqs[0]
            meas_vector[1] = results[0]
        else:
            meas_error = 1

        if meas_error == 1:
            print("Error, measurement failed")

        meas_vpp[0] = (np.sqrt(meas_vector[0][2]) / NUM_ACQ_POINTS) * 4
        meas_vpp[1] = (np.sqrt(meas_vector[1][2]) / NUM_ACQ_POINTS) * 4

        c1vectors.append([meas_freq[0], meas_vector[0][0], meas_vector[0][1], meas_vector[0][2], meas_vpp[0]])
        c2vectors.append([meas_freq[1], meas_vector[1][0], meas_vector[1][1], meas_vector[1][2], meas_vpp[1]])


def plot_chart():
    global freq1, vpp1, real1, imag1, freq2, vpp2, vpp, real2, imag2, deg1, deg2, deg, real, imag, s11
    freq1 = [row[0] for row in c1vectors]
    vpp1 = [row[4] for row in c1vectors]
    real1 = [row[1] for row in c1vectors]
    imag1 = [row[2] for row in c1vectors]

    freq2 = [row[0] for row in c2vectors]
    vpp2 = [row[4] for row in c2vectors]
    real2 = [row[1] for row in c2vectors]
    imag2 = [row[2] for row in c2vectors]

    deg1 = [math.degrees(math.atan2(y, x)) for (x, y) in zip(real1, imag1)]
    deg2 = [math.degrees(math.atan2(y, x)) for (x, y) in zip(real2, imag2)]

    gain = [20 * np.log10(v2 / v1) for (v1, v2) in zip(vpp1, vpp2)]
    deg = [x2 - x1 for (x1, x2) in zip(deg1, deg2)]

    # plot the data
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=freq1, y=gain, name="Gain"), secondary_y=False)
    fig.add_trace(go.Scatter(x=freq1, y=deg, name="Phase"), secondary_y=True)

    # format the chart
    fig.update_layout(title_text="Frequency Response", xaxis_type="log", title_font=dict(size=30))
    fig.update_xaxes(title_text="Frequency (Hz)", tickfont=dict(size=20), title_font=dict(size=20))
    fig.update_yaxes(title_text="Gain (dB)", secondary_y=False, tickfont=dict(size=20), title_font=dict(size=20))
    fig.update_yaxes(title_text="Phase (deg)", secondary_y=True, tickfont=dict(size=20), title_font=dict(size=20))
    # set figure background color to white
    fig.update_layout(plot_bgcolor="white")
    # set grid line color to dark grey
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='DarkGrey')  # vert grid lines

    fig.update_yaxes(ticks="inside", ticklen=10, minor=dict(ticks="inside", ticklen=6))
    # add minor x ticks
    fig.update_xaxes(ticks="inside", ticklen=10,
                     minor=dict(ticks="inside", ticklen=6, showgrid=True, gridwidth=1, gridcolor='LightGrey'))
    # set plot line thickness
    fig.update_traces(line=dict(width=3), marker_size=12)
    # display the finished chart
    fig.show()


def print_welcome():
    print("\n*** Welcome to Audio Analyzer for MXO 4 ***")


# ---------------------------------------
if __name__ == '__main__':
    print_welcome()
    if open_instrument():
        pass
    else:
        print("Instrument not found!")
        exit()

    do_scan()

    plot_chart()

    close_instrument()

    print("Done!")
