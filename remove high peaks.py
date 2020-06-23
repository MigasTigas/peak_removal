import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

date_form = DateFormatter("%H:%M:%S")  # Xlabel for the plot_values
real_peaks = [43, 89, 234, 280, 370, 417, 418, 456, 500, 741, 789, 837, 884, 932, 980, 1028, 1074]  # All added peaks


def plot_values(dates=None, values=None, dates_indexes=None, values_indexes=None):
    """
    This function is to plot the water flow accordingly to the timestamps of measuring

    :param dates: array-like of pydatetimes
    :param values: array-like of floats
    :param dates_indexes: peak filtered dates
    :param values_indexes: peak filtered values
    :return:
    """

    global dataset

    plt.cla()  # Clears previous plots

    fig = plt.figure(1, figsize=(7, 7))
    ax1 = fig.add_subplot(111)

    ax1.plot(dates, values, 'c', label="Water flow")  # Plotting of all values
    ax1.plot(dates_indexes, values_indexes, "xr", label="peaks")  # Plotting of all peaks
    ax1.xaxis.set_major_formatter(date_form)

    # ax2 = ax1
    # ax1.plot(dates_indexes, values_indexes, "xr", label="peaks")  # Plotting of all peaks
    plt.xlabel("Timestamp of measure")
    plt.ylabel("Water flow (m\u00b3)")
    plt.legend()
    plt.title("Water flow over time")
    plt.show()


def plot_slopes():
    """
        This function is to plot the slope that the water flow creates
    """
    plt.plot(slopes, label="slopes")
    plt.hlines(t, 0, len(values), linestyles="dashed", label="Threshold")
    plt.hlines(-t, 0, len(values), linestyles="dashed", label="Threshold")
    plt.plot(real_peaks, slopes[real_peaks], "*c", label="Peaks")
    plt.plot(peaks, slopes[peaks], "^r", label="Detected peaks")

    plt.legend()
    plt.title("The slopes of the flow measures")
    plt.show()


def plot_flow():
    """
    This function plots the differences between each point
    """

    plt.plot(flow, label="Flow diff")
    plt.plot(real_peaks, flow[real_peaks], "*c", label="Peaks")
    plt.plot(peaks, flow[peaks], "^r", label="Detected peaks")

    plt.legend()
    plt.title("The differences in the flow measures")
    plt.show()


# Started by reading the .csv and parse the dates, followed by splitting into two numpy arrays:
dataset = pd.read_csv("https://raw.githubusercontent.com/MigasTigas/peak_removal/master/peak_simulation.csv",
                      parse_dates=['date'])
dataset = dataset.sort_values(by=['date']).reset_index(drop=True)

# Then applied the (flow[i+1] - flow[i]) / (time[i+1] - time[i]) to the whole dataset:
values = dataset['value'].values
dates = dataset['date'].values

# Create the diffs
flow = np.diff(values)
time = np.diff(dates).tolist()
time = np.divide(time, np.power(10, 9))

slopes = np.divide(flow, time)  # (flow[i+1] - flow[i]) / (time[i+1] - time[i])
slopes = np.insert(slopes, 0, 0, axis=0)  # Since we "lose" the first index, this one is 0, just for alignments

# And finally to detect the peaks we reduced the data to rolling windows of x seconds each. That way we can detect them easily:
size = len(dataset)
rolling_window = []
rolling_window_indexes = []
RW = []
RWi = []
window_size = 240  # Seconds

dates = [i.to_pydatetime() for i in dataset['date']]
dates = np.array(dates)

# create the rollings windows
for line in range(size):
    limit_stamp = dates[line] + datetime.timedelta(seconds=window_size)
    for subline in range(line, size, 1):
        if dates[subline] <= limit_stamp:

            rolling_window.append(slopes[subline])  # Values of the slopes
            rolling_window_indexes.append(subline)  # Indexes of the respective values

        else:

            RW.append(rolling_window)
            if line != size:  # To prevent clearing the last rolling window
                rolling_window = []

            RWi.append(rolling_window_indexes)
            if line != size:
                rolling_window_indexes = []
            break
else:
    # To get the last rolling window since it breaks before append
    RW.append(rolling_window)
    RWi.append(rolling_window_indexes)


# After getting all rolling windows we start the fun:
t = 0.3  # Threshold
peaks = []

for index, rollWin in enumerate(RW):
    if rollWin[0] > t: # If the first value is greater of threshold
        top = rollWin[0] # Sets as a possible peak
        bottom = np.min(rollWin) # Finds the minimum of the peak

        if bottom < -t: # If less than the negative threshold
            bottomIndex = int(np.argmin(rollWin)) # Find it's index

            for peak in range(0, bottomIndex, 1): # Appends all points between the first index of the rolling window until the bottomIndex
                peaks.append(RWi[index][peak])


# PLOTTING
plot_values(dates, values, dates[peaks], values[peaks]) # The real dataset and the peaks
plot_slopes() # The slopes and the peaks
plot_flow() # The differences between points
