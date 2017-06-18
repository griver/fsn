# coding=utf-8

import json
import matplotlib.pyplot as plt
import numpy as np

def get_json_data(filename):
    """
    Загружает данные испытаний из указанного файла
    """
    data = None
    with open(filename, "r") as the_file:
        data = json.load(the_file)

    return data


def trials_median(trials):
    """
    args: 
    trials - list(learning_1, learning_2, .. learning_k), where each learning_i is a list of tials results during i-th learning process. 
      
    return 
    list of median values.    
    """
    trials = trials if isinstance(trials, np.ndarray) else np.array(trials) 

    x_len, y_len = trials.shape
    sorted_trials = np.sort(trials, axis=0)

    if x_len % 2 :
        get_median = lambda i: sorted_trials[x_len/2][i]
    else:
        get_median = lambda i: (sorted_trials[x_len/2][i] + sorted_trials[x_len/2 - 1][i]) / 2.0

    median = [get_median(i) for i in xrange(y_len)]
    return median


def black_gray_plot(gray_plots, black_plot, x_label="trials", y_label="path_length", x_range=None, y_range=None):
    """
    Рисует множество графиков серым и один основной черной жирной линией. Общая функция для отображения хода отдельных обучений и 
    их среднего/медианы. 
    """   
    x_len =  len(gray_plots[0]) if len(gray_plots) else len(black_plot)
    x_axis = range(1, x_len + 1)
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)

    if x_range is not None:
        ax1.set_xlim(*x_range)
    if y_range is not None:
        ax1.set_ylim(*y_range)

    for plot in gray_plots:
        ax1.plot(x_axis, plot, color='gray')

    ax1.plot(x_axis, black_plot, color='black', lw=3.0)

    if x_label is not None:
        ax1.set_xlabel(x_label)       
    if y_label is not None:
        ax1.set_ylabel(y_label)          
   
    plt.show()


def average_values_correlation(x_lists, y_lists, x_label="", y_label="", x_range=None, y_range=None):
    x_points = []
    y_points = []
    min_len = min(len(x_lists), len(y_lists))
    for i in xrange(min_len):
        x_points.append(np.average(x_lists[i]))
        y_points.append(np.average(y_lists[i])) 

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)

    ax1.plot(x_points, y_points,  'ob')

    maxx = max(max(x_points), max(y_points))
    maxx += maxx / 10

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)

    if x_range is None:
        ax1.set_xlim(0, maxx)
    else:
        ax1.set_xlim(*x_range)
    
    if y_range is None:
        ax1.set_ylim(0, maxx)
    else:
        ax1.set_ylim(*y_range)

    plt.show()

