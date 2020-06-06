import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def spacing(dates, values):
    '''
    This is one of the attemps to remove outlier peaks, using MAD (Median Absolute Deviation)
    It first calculates all spacings between two consecutive values and then creates the median between a rooling window
    where it's amplified by a threshold to see where the values are higher than the median, detecting, suposedly,
    the peak that is wrong
    :param dates: Timestamps, in datetime
    :param values: Water flow values, in float
    '''
    size = len(dates)

    def myMAD(x):
        '''
        A simple Median Absolute Deviation
        :param x: set of values
        :return MAD: set of values with MAD calculated
        '''
        med = np.median(x)
        x = abs(x - med)
        MAD = np.median(x)
        return MAD

    value_spacing = np.zeros(size)
    for line in range(1, size, 1):
        value_spacing[line - 1] = float((values[line] - values[line - 1]))

    window_size = 300  # Seconds
    threshold = 1.5
    rolling_window = []
    medians = []
    peaks = []

    # Calculating the median of the rolling window of the dataset
    for line in range(0, size, 1):
        # Find the current timestamp and set a limit stamp to create the rolling window
        limit_stamp = dates[line] + datetime.timedelta(seconds=window_size)
        for subline in range(line, size, 1):
            if dates[subline] < limit_stamp:
                rolling_window.append(value_spacing[subline])
            else:
                mad_RW = myMAD(rolling_window)  # Median Absolute deviation
                median_RW = np.median(rolling_window)  # Normal Median

                # Insert into list, the median with the amplifification (threshold)
                medians.append(median_RW + threshold * mad_RW)
                rolling_window = []
                break

    # Find peaks where the value is higher than his median
    for i, median in enumerate(medians):
        if value_spacing[i] > median:
            peaks.append(i)

    plt.cla()
    plt.plot(value_spacing, "g", label="Spacing values")  # The spacing of the values, in blue
    plt.plot(548, value_spacing[548], "ob", label="Actual outlier peak")  # The actual outlier peak, in blue
    plt.plot(peaks, value_spacing[peaks], "xr", label="Peaks detected")  # The peaks detected, in red crosses
    plt.plot(medians, "y", label="Median")  # The medians of the values, in yellow
    plt.legend()

    plt.show()


def diff_percent(values):
    '''
    This is a simple detection, it calculates the percentual difference between two values and if its higher than
    1 (meaning that it doubles at least) states as a peak
    :param values: Water flow values, in float
    '''

    size = len(values)
    peaks = []
    diff_points = []
    for line in range(1, size, 1):
        diff = ((values[line] - values[line - 1]) / values[line - 1]) * 100
        diff_points.append(diff)
        if np.abs(diff) > 100:
            peaks.append(line)

    plt.cla()
    plt.plot(values, "g", label="Real values")  # The real values, in green
    plt.plot(diff_points, "y", label="Percentual differences") # The percentual differences, in yellow
    plt.plot(549, values[549], "ob", label="Actual outlier peak")  # The actual outlier peak, in blue
    plt.plot(peaks, values[peaks], "xr", label="Peaks detected")  # The peaks detected, in red crosses
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # Fetch dataset
    dataset = pd.read_csv("datasets/dataset_simple_example.csv")
    dataset = dataset.sort_values(by=['date']).reset_index(drop=True).to_numpy() # Sort and convert to numpy array

    # Split into 2 arrays
    values = [float(i[1]) for i in dataset]  # Flow values, in float
    values = np.array(values)

    dates = [datetime.datetime.strptime(i[0], '%Y-%m-%d %H:%M:%S') for i in dataset]  # Timestamps, in datetime
    dates = np.array(dates)

    plt.figure(num='Value spacing detection')
    spacing(dates, values) # Detecting via value spacing (value[1] - value[0]) and medians
    plt.figure(num='Percentual difference detection')
    diff_percent(values)  # Detecting via percentual difference

