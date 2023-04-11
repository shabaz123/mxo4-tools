# -*- coding: utf-8 -*-
"""
Spectrum Plotting Tool for MXO 4
main.py - rev 1 - shabaz
@author: shabaz
"""
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import cmath
import numpy as np
import math
import csv
import json
import time

# configuration parameters

# test parameters
csvname = 'm1b.csv'
limit_fname = "cispr32_1.json"  # file containing frequency limits

# charting parameters
DO_LOG = 1  # set to 1 to plot the frequency on log scale
DO_LIMIT_LINES = 1  # set to 1 to plot frequency limit lines


# other constants



# globals
hkeys = []  # stores header keys
hvalues = []  # stores header values
hdata = []  # stores header data (two columns from CSV file)
sdata = []  # stores spectrum data (two columns from CSV file)
sfreqs = [] # stores spectrum frequencies, which are contents of first column extracted from sdata
svalues = []  # stores spectrum values, which are contents of second column extracted from sdata


# functions

# read CSV file into sdata
def read_csv():
    global hkeys, hvalues, hdata
    global sdata, svalues, sfreqs
    fdata=[]
    with open(csvname, 'r') as f:
        reader = csv.reader(f)
        fdata = list(reader)
    for i in range(len(fdata)):
        if fdata[i][0] == 'TIME':
            hdata = fdata[:i-1]
            sdata = fdata[i+1:]
            break

    sfreqs = [row[0] for row in sdata]
    svalues = [row[1] for row in sdata]

    hkeys = [row[0] for row in hdata]
    hvalues = ["0", "0", "0"]
    for i in range(3, len(hdata)):
        hvalues.append(hdata[i][1])



def get_config_key_value(key):
    for i in range(len(hkeys)):
        if hkeys[i] == key:
            return hvalues[i]
    return None


def plot_chart():
    fstart = get_config_key_value('FrequencyStart')
    fstop = get_config_key_value('FrequencyStop')
    # plot the data
    fig = make_subplots()

    # find first value which is greater than fstart
    for i in range(len(sfreqs)):
        if float(sfreqs[i]) > float(fstart):
            break
    # find last value which is less than fstop
    for j in range(len(svalues)-1, 0, -1):
        if float(sfreqs[j]) < float(fstop):
            break
    # extract the data between fstart and fstop
    print(f"fstart={fstart}, fstop={fstop}")
    print(f"i={i}, j={j}")

    svalues1_str = svalues[i:j]
    sfreqs1_str = sfreqs[i:j]

    svalues1 = []
    sfreqs1 = []
    for i in range(len(svalues1_str)):
        svalues1.append(float(svalues1_str[i]))
        sfreqs1.append(float(sfreqs1_str[i]))

    fig.add_trace(go.Scatter( x=sfreqs1, y=svalues1, name="Spectrum"))

    # add frequency limit lines
    lvalues=[]
    lfreqs=[]
    if DO_LIMIT_LINES > 0:
        # read frequency limits from file
        with open(limit_fname) as json_file:
            limits = json.load(json_file)
        for i in range(0, limits['total_plots']):
            lvalues = limits[f"yval{i+1}"]  # get the y values for the limit line
            lfreqsM = limits[f"freq{i+1}"]  # get the MHz freq values for the limit line
            lfreqs = [j * 1000000 for j in lfreqsM]  # convert to Hz
            fig.add_trace(go.Scatter(x=lfreqs, y=lvalues, name=limits[f"name{i+1}"],))


    # format the chart
    if DO_LOG > 0:
        fig.update_layout(xaxis_type="log")
    fig.update_layout(title_text="Spectral Analysis", title_font=dict(size=30))
    fig.update_xaxes(title_text="Frequency (Hz)", tickfont=dict(size=20), title_font=dict(size=20))
    if get_config_key_value("ViewUnit") == "LEVEL_DBM":
        fig.update_yaxes(title_text="Power (dBm)", tickfont=dict(size=20), title_font=dict(size=20))
    if get_config_key_value("ViewUnit") == "LEVEL_DB_MIKRO_V":
        fig.update_yaxes(title_text="Voltage (dBuV)", tickfont=dict(size=20), title_font=dict(size=20))
    # set figure background color to white
    fig.update_layout(plot_bgcolor="white")
    # set grid line color to dark grey
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='DarkGrey')  # vert grid lines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='DarkGrey')  # horiz grid lines

    fig.update_yaxes(ticks="inside", ticklen=10, minor=dict(ticks="inside", ticklen=6))
    # add minor x ticks
    fig.update_xaxes(ticks="inside", ticklen=10,
                     minor=dict(ticks="inside", ticklen=6, showgrid=True, gridwidth=1, gridcolor='LightGrey'))
    # set plot line thickness
    fig.update_traces(line=dict(width=3), marker_size=12)

    # display the finished chart
    fig.show()


def print_welcome():
    print("\n*** Welcome to Spectrum Tool for MXO 4 ***")


# ---------------------------------------
if __name__ == '__main__':
    print_welcome()

    read_csv()
    plot_chart()


    print("Done!")
