import numpy as np
import pydub
import librosa
import os
import shutil
import urllib
import zipfile
import glob
import matplotlib
import matplotlib.pyplot as plt
from operator import add
import APIs as api
import tensorflow as tf
import pandas as pd
import APIs as api
from sys import getsizeof

import tensorflow as tf
import numpy as np
from operator import add

import seaborn as sb
import time

def add_subplot_axes(ax, position):
    box = ax.get_position()
    position_display = ax.transAxes.transform(position[0:2])
    position_fig = plt.gcf().transFigure.inverted().transform(position_display)
    x = position_fig[0]
    y = position_fig[1]
    return plt.gcf().add_axes([x, y, box.width * position[2], box.height * position[3]], axisbg='w')


def plot_clip_overview(clip, ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax_waveform = add_subplot_axes(ax, [0.0, 0.7, 1.0, 0.3])
    ax_spectrogram = add_subplot_axes(ax, [0.0, 0.0, 1.0, 0.7])

    with clip.audio as audio:
        ax_waveform.plot(np.arange(0, len(audio.raw)) / float(api.Clip.RATE), audio.raw)
        ax_waveform.get_xaxis().set_visible(False)
        ax_waveform.get_yaxis().set_visible(False)
        ax_waveform.set_title('{0} \n {1}'.format(clip.category, clip.filename), {'fontsize': 8}, y=1.03)

        librosa.display.specshow(clip.logamplitude, sr=api.Clip.RATE, x_axis='time', y_axis='mel', cmap='RdBu_r')
        ax_spectrogram.get_xaxis().set_visible(False)
        ax_spectrogram.get_yaxis().set_visible(False)


def plot_single_clip(clip):
    col_names = list('MFCC_{}'.format(i) for i in range(np.shape(clip.mfcc)[1]))
    MFCC = pd.DataFrame(clip.mfcc[:, :], columns=col_names)

    f = plt.figure(figsize=(10, 6))
    ax = f.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    ax_mfcc = add_subplot_axes(ax, [0.0, 0.0, 1.0, 0.75])
    ax_mfcc.set_xlim(-400, 400)
    ax_zcr = add_subplot_axes(ax, [0.0, 0.85, 1.0, 0.05])
    ax_zcr.set_xlim(0.0, 1.0)

    plt.title('Feature distribution across frames of a single clip ({0} : {1})'.format(clip.category, clip.filename), y=1.5)
    sb.boxplot(data=MFCC, orient='h', order=list(reversed(MFCC.columns)), ax=ax_mfcc)
    sb.boxplot(data=pd.DataFrame(clip.zcr, columns=['ZCR']), orient='h', ax=ax_zcr)
    plt.show()



def plot_single_feature_one_clip(feature, title, ax):
    sb.despine()
    ax.set_title(title, y=1.10)
    sb.distplot(feature, bins=20, hist=True, rug=False,
                hist_kws={"histtype": "stepfilled", "alpha": 0.5},
                kde_kws={"shade": False},
                color=sb.color_palette("muted", 4)[2], ax=ax)


def plot_single_feature_all_clips(feature, title, ax):
    sb.despine()
    ax.set_title(title, y=1.03)
    sb.boxplot(feature, vert=False, orient='h', order=list(reversed(feature.columns)), ax=ax)


def plot_single_feature_aggregate(feature, title, ax):
    sb.despine()
    ax.set_title(title, y=1.03)
    sb.distplot(feature, bins=20, hist=True, rug=False,
                hist_kws={"histtype": "stepfilled", "alpha": 0.5},
                kde_kws={"shade": False},
                color=sb.color_palette("muted", 4)[1], ax=ax)


def generate_feature_summary(dataset, category, clip, coefficient):
    title = "{0} : {1}".format(dataset[category][clip].category, dataset[category][clip].filename)
    MFCC = pd.DataFrame()
    aggregate = []
    for i in range(0, len(dataset[category])):
        MFCC[i] = dataset[category][i].mfcc[:, coefficient]
        aggregate = np.concatenate([aggregate, dataset[category][i].mfcc[:, coefficient]])

    f = plt.figure(figsize=(14, 12))
    f.subplots_adjust(hspace=0.6, wspace=0.3)

    ax1 = plt.subplot2grid((3, 3), (0, 0))
    ax2 = plt.subplot2grid((3, 3), (1, 0))
    ax3 = plt.subplot2grid((3, 3), (0, 1), rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)

    ax1.set_xlim(0.0, 0.5)
    ax2.set_xlim(-100, 250)
    ax4.set_xlim(-100, 250)

    plot_single_feature_one_clip(dataset[category][clip].zcr, 'ZCR distribution across frames\n{0}'.format(title), ax1)
    plot_single_feature_one_clip(dataset[category][clip].mfcc[:, coefficient],
                                 'MFCC_{0} distribution across frames\n{1}'.format(coefficient, title), ax2)

    plot_single_feature_all_clips(MFCC, 'Differences in MFCC_{0} distribution\nbetween clips of {1}'.format(coefficient,
                                                                                    dataset[
                                                                                                                category][
                                                                                                                clip].category),
                                  ax3)

    plot_single_feature_aggregate(aggregate,'Aggregate MFCC_{0} distribution\n(bag-of-frames across all clips\nof {1})'.format(
                                      coefficient, dataset[category][clip].category), ax4)
    plt.show()



    clips_50 = api.load_dataset('ESC-50')

    # all_recordings = glob.glob('ESC-10/*/*.ogg')
    # clip = Clip(all_recordings[random.randint(0, len(all_recordings) - 1)])
    #
    # with clip.audio as audio:
    #     plt.subplot(2, 1, 1)
    #     plt.title('{0} : {1}'.format(clip.category, clip.filename))
    #     plt.plot(np.arange(0, len(audio.raw)) / 44100.0, audio.raw)
    #     plt.subplot(2, 1, 2)
    #     librosa.display.specshow(clip.logamplitude, sr=44100, x_axis='frames', y_axis='linear', cmap='RdBu_r')


    # direct to the local dir and run this in terminal:
    # $ tensorboard --logdir=logs

    # categories = 5
    # clips_shown = 1
    # f, axes = plt.subplots(categories, clips_shown, figsize=(clips_shown * 2, categories * 2), sharex=True, sharey=True)
    # f.subplots_adjust(hspace=0.35)

    # for c in range(0, categories):
    #     for i in range(0, clips_shown):
    #         plot_clip_overview(clips_10[c][i], axes[c, i])
    # plt.show()

    # plot_single_clip(clips_10[2][0])

    # generate_feature_summary(clips_10, 1, 0, 1)