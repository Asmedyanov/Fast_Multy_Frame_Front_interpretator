from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks


def open_xlsx(fname):
    """
    the function read the info file of the experiment
    :param fname: file name
    :return:
    pandas data frame
    """
    data = pd.read_excel(fname)
    data = data.set_index('Parameter')
    return data


def open_rtv(fname):
    """
    the function read the binary file of the fast-frame xrapid camera
    :param fname: file name
    :return:
    numpy array (4,1024,1360)
    4 frames
    """
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
    """
    the function read the waveform file *.csv:
    current,synchro:camera and maxwell, voltage divider
    :param fname: file name
    :param Rogovski_ampl: coefficient to transform voltage from the Rogovski coil to Amper
    :param Rogovski_conv: the number of points to smooth the current
    :return:
    {
        'time': current_time,
        'current': current_amp,
        'peaks': peak_times
    }
    """
    waveform = pd.read_csv(fname)
    '''plt.plot(1.0e6*waveform['s'],waveform['Volts']/np.abs(waveform['Volts']).max(),label='Current')
    plt.plot(1.0e6*waveform['s.1'],waveform['Volts.1']/np.abs(waveform['Volts.1']).max(),label='Main trig')
    plt.plot(1.0e6*waveform['s.2'],waveform['Volts.2']/np.abs(waveform['Volts.2']).max(),label='4Quick trig')
    plt.plot(1.0e6*waveform['s.3'],waveform['Volts.3']/np.abs(waveform['Volts.3']).max(),label='Tektronix')
    plt.xlabel('t, us')
    plt.legend()
    plt.show()'''
    sinc_time = waveform['s.1'].values * 1.0e6
    sinc_volt = np.abs(np.gradient(waveform['Volts.1']))
    if sinc_volt.max() < 10.0 * sinc_volt.mean():
        sinc_volt = np.abs(np.gradient(waveform['Volts.2']))
    peaks = find_peaks(sinc_volt[:sinc_volt.size // 2], prominence=0.02, distance=50)[0]
    peaks = peaks[-16:]
    peak_times = sinc_time[peaks]
    current_volt = waveform['Volts'].values
    '''plt.plot(sinc_time, sinc_volt)
    plt.plot(sinc_time, current_volt)
    plt.plot(sinc_time[peaks], sinc_volt[peaks],'o')

    plt.show()'''
    current_amp = current_volt * Rogovski_ampl
    n_conv = Rogovski_conv
    a_conv = np.ones(n_conv) / float(n_conv)
    current_amp = np.convolve(current_amp, a_conv, mode='same')[n_conv // 2:-n_conv // 2 - 1]
    current_time = np.convolve(sinc_time, a_conv, mode='same')[n_conv // 2:-n_conv // 2 - 1]
    zero_ind = np.argwhere(current_time < 0).max()
    noise = current_amp[:zero_ind]
    current_amp -= noise.mean()
    noise_ample = np.abs(noise - noise.min())
    current_start = np.argwhere(np.abs(current_amp) > 0.8 * np.max(noise_ample)).min()
    main_shift = current_time[current_start]
    peak_times -= main_shift
    current_time -= main_shift
    plt.plot(current_time, current_amp/current_amp.max())
    plt.plot(sinc_time, sinc_volt/sinc_volt.max())
    plt.plot(sinc_time[peaks], sinc_volt[peaks]/sinc_volt.max(),'o')
    plt.plot(peak_times, current_amp[peaks]/current_amp.max(),'o')
    plt.show()
    ret = {
        'time': current_time,
        'current': current_amp,
        'peaks': peak_times
    }
    return ret


def open_folder():
    """
    The function loads the data of experiment from file dialog
    the experiment folder includes:
    'info.xlsx' file with scalar data of experiment
    'shot*.csv' file with waveforms
    'before.rtv' bin file with images from xrapid came
    :return:
    dict of data
    """
    folder_name = filedialog.askdirectory(
        initialdir='C:/Users/User/OneDrive - Technion/UEWE/Foils/Butterfly/multiframe')
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
