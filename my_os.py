from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks


def open_xlsx(fname):
    #data = pd.read_excel(fname, index_col='Parameter')
    data = pd.read_excel(fname)
    data = data.set_index('Parameter')
    return data


def open_rtv(fname):
    file = open(fname, 'rb')
    n = 1024 * 1360
    file_array = np.fromfile(file, dtype='uint16', offset=0x2000, count=n * 4).reshape((4, 1024, 1360))

    ar_right = np.copy(file_array[1::2, :, :1360 // 2])
    ar_left = np.copy(file_array[1::2, :, 1360 // 2:])
    file_array[1::2, :, :1360 // 2] = ar_left
    file_array[1::2, :, 1360 // 2:] = ar_right

    image_array = np.copy(file_array)
    file.close()
    return image_array


def open_csv(fname, Rogovski_ampl, Rogovski_conv):
    waveform = pd.read_csv(fname)
    sinc_time = waveform['s.1'].values * 1.0e6
    sinc_volt = np.abs(np.gradient(waveform['Volts.1']))
    if sinc_volt.max()<10.0*sinc_volt.mean():
        sinc_volt = np.abs(np.gradient(waveform['Volts.2']))
    peaks = find_peaks(sinc_volt[:sinc_volt.size//2], prominence=0.1)[0]
    peaks = peaks[-8:]
    peak_times = sinc_time[peaks]
    current_volt = waveform['Volts'].values
    current_amp = current_volt * Rogovski_ampl
    n_conv = Rogovski_conv
    a_conv = np.ones(n_conv) / float(n_conv)
    current_amp = np.convolve(current_amp, a_conv, mode='same')[n_conv // 2:-n_conv // 2 - 1]
    current_time = np.convolve(sinc_time, a_conv, mode='same')[n_conv // 2:-n_conv // 2 - 1]
    zero_ind = np.argwhere(current_time < 0).max()
    noise = current_amp[:zero_ind]
    current_amp -= noise.mean()
    noise_ample = np.std(noise)
    current_start = np.argwhere(np.abs(current_amp) > 2.2 * np.max(noise)).min()
    main_shift = current_time[current_start]
    peak_times -= main_shift
    current_time -= main_shift
    ret = {
        'time': current_time,
        'current': current_amp,
        'peaks': peak_times
    }
    return ret


def open_folder():
    folder_name = filedialog.askdirectory(initialdir='C:/Users/User/Butterfly_processing/Nikita_Processing')
    current_dir = os.curdir
    os.chdir(folder_name)
    files_data = dict()
    files_data['info'] = open_xlsx('info.xlsx')
    for fname in os.listdir():
        if fname.split('.')[-1] == 'rtv':
            data = open_rtv(fname)
            if fname.split('.')[0] == 'before':
                files_data['before'] = data
            else:
                files_data['shot'] = data
            continue
        if fname.split('.')[-1] == 'csv':
            files_data['waveform'] = open_csv(fname, -files_data['info']['Value']['Rogovski_ampl'],
                                              files_data['info']['Value']['Rogovski_conv'])
            continue
        '''if fname.split('.')[-1] == 'xlsx':
            files_data[fname] = open_xlsx(fname)
            continue'''
    pass
    return files_data


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
    current_amp -= noise.mean()
    noise_ample = np.std(noise)
    current_start = np.argwhere(np.abs(current_amp) > 2.2 * np.max(noise)).min()

    # plt.plot(current_time[current_start], current_amp[current_start] * 1.0e-3, 'o')
    main_shift = current_time[current_start]
    peak_times -= main_shift
    current_time -= main_shift
    # plt.plot(current_time, current_amp * 1.0e-3)
    # plt.plot(sinc_time+main_shift, current_volt)
    # plt.show()
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
