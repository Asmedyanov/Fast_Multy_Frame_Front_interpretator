from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks


def open_images():
    # fname = filedialog.askopenfilename(title="Open image before", filetypes=[("images", ".rtv")])
    fname = 'before.rtv'
    file = open(fname, 'rb')
    n = 1024 * 1360
    file_array = np.fromfile(file, dtype='uint16', offset=0x2000, count=n * 4).reshape((4, 1024, 1360))

    ar_right = np.copy(file_array[1::2, :, :1360 // 2])
    ar_left = np.copy(file_array[1::2, :, 1360 // 2:])
    file_array[1::2, :, :1360 // 2] = ar_left
    file_array[1::2, :, 1360 // 2:] = ar_right

    before_array = np.copy(file_array)
    file.close()
    # fname = filedialog.askopenfilename(title="open image shot", filetypes=[("images", ".rtv")])
    fname = 'shot25.rtv'
    file = open(fname, 'rb')
    file_array = np.fromfile(file, dtype='uint16', offset=0x2000, count=n * 4).reshape((4, 1024, 1360))

    ar_right = np.copy(file_array[1::2, :, :1360 // 2])
    ar_left = np.copy(file_array[1::2, :, 1360 // 2:])
    file_array[1::2, :, :1360 // 2] = ar_left
    file_array[1::2, :, 1360 // 2:] = ar_right

    shot_array = np.copy(file_array)

    fname_wf = 'shot25.csv'
    waveform = pd.read_csv(fname_wf)
    sinc_time = waveform['s.1'].values * 1.0e6
    sinc_volt = np.abs(np.gradient(waveform['Volts.1']))
    peaks = find_peaks(sinc_volt, prominence=0.1)[0]
    peaks = peaks[-8:]
    peak_times = sinc_time[peaks]
    current_volt = waveform['Volts.2'].values
    current_amp = current_volt * 39188.5
    n_conv = 50
    a_conv = np.ones(n_conv) / float(n_conv)
    current_amp = np.convolve(current_amp, a_conv, mode='same')[n_conv // 2:-n_conv // 2 - 1]
    current_time = np.convolve(sinc_time, a_conv, mode='same')[n_conv // 2:-n_conv // 2 - 1]
    zero_ind = np.argwhere(current_time < 0).max()
    noise = current_amp[:zero_ind]
    current_amp-=noise.mean()
    noise_ample = np.std(noise)
    current_start = np.argwhere(np.abs(current_amp) > 2.2*np.max(noise)).min()

    #plt.plot(current_time[current_start], current_amp[current_start] * 1.0e-3, 'o')
    main_shift = current_time[current_start]
    peak_times -= main_shift
    current_time -= main_shift
    #plt.plot(current_time, current_amp * 1.0e-3)
    # plt.plot(sinc_time+main_shift, current_volt)
    #plt.show()
    pass
    '''fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(file_array[0])
    ax[0, 1].imshow(file_array[1])
    ax[1, 0].imshow(file_array[2])
    ax[1, 1].imshow(file_array[3])
    plt.show()'''

    fname = fname.split('/')[-1].split('.')[0]
    try:
        os.mkdir('Streack_processing')
    except:
        pass
    return fname, before_array, shot_array, peak_times, current_time, current_amp
