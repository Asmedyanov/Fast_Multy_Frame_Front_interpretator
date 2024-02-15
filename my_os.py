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


def open_csv(fname, Rogovski_ampl, Rogovski_conv, Inductance):
    """
    the function read the waveform file *.csv:
    current,synchro:camera and maxwell, voltage divider
    :param Inductance:
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
    '''plt.plot(1.0e6 * waveform['s'], waveform['Volts'] / np.abs(waveform['Volts']).max(), label='Current')
    plt.plot(1.0e6 * waveform['s.1'], waveform['Volts.1'] / np.abs(waveform['Volts.1']).max(), label='Main trig')
    plt.plot(1.0e6 * waveform['s.2'], waveform['Volts.2'] / np.abs(waveform['Volts.2']).max(), label='4Quick trig')
    plt.plot(1.0e6 * waveform['s.3'], waveform['Volts.3'] / np.abs(waveform['Volts.3']).max(), label='Tektronix')
    plt.xlabel('t, us')
    plt.legend()
    plt.show()'''
    sinc_time = waveform['s.1'].values * 1.0e6
    sinc_volt = np.abs(np.gradient(waveform['Volts.1']))
    if sinc_volt.max() < 10.0 * sinc_volt.mean():
        sinc_volt = np.abs(np.gradient(waveform['Volts.2']))
    peaks = find_peaks(sinc_volt[:sinc_volt.size // 2], prominence=0.05, distance=20)[0]
    peaks = peaks[-16:]
    peak_times = sinc_time[peaks]
    current_volt = waveform['Volts'].values
    volt_volt = waveform['Volts.3'].values * 1.0e3
    plt.plot(sinc_time, sinc_volt)
    plt.plot(sinc_time, current_volt)
    plt.plot(sinc_time[peaks], sinc_volt[peaks],'o')

    plt.show()
    current_amp = current_volt * Rogovski_ampl
    n_conv = Rogovski_conv
    a_conv = np.ones(n_conv) / float(n_conv)
    current_amp = np.convolve(current_amp, a_conv, mode='same')[n_conv // 2:-n_conv // 2 - 1]
    volt_volt = np.convolve(volt_volt, a_conv, mode='same')[n_conv // 2:-n_conv // 2 - 1]
    current_time = np.convolve(sinc_time, a_conv, mode='same')[n_conv // 2:-n_conv // 2 - 1]
    zero_ind = np.argwhere(current_time < 0).max()
    noise = volt_volt[:zero_ind]
    noise_current = current_amp[:zero_ind]
    volt_volt -= noise.mean()
    current_amp -= noise_current.mean()

    L0 = Inductance
    U_res = volt_volt - L0 * np.gradient(current_amp) / np.gradient(current_time).mean() * 1.0e6
    U_res = np.convolve(U_res, a_conv, mode='same')  # [n_conv // 2:-n_conv // 2 - 1]
    Power = U_res * current_amp
    Resistance = np.where((current_amp > 2.0e-1 * current_amp.max()) & (U_res > 0), U_res / current_amp, 0)
    Power = np.convolve(Power, a_conv, mode='same')  # [n_conv // 2:-n_conv // 2 - 1]
    Resistance = np.convolve(Resistance, a_conv, mode='same')  # [n_conv // 2:-n_conv // 2 - 1]

    noise_ample = np.abs(noise - noise.min())
    current_start = np.argwhere(np.abs(volt_volt) > 0.8 * np.max(noise_ample)).min()
    main_shift = current_time[current_start]
    peak_times -= main_shift
    current_time -= main_shift
    #plt.plot(current_time, volt_volt)
    current_time_to_approx_index = np.argwhere(((current_time >= 0) & (current_time < 1)))[:,0]
    voltage_time_to_approx = current_time[current_time_to_approx_index]
    voltage_to_approx = volt_volt[current_time_to_approx_index]
    #plt.plot(voltage_time_to_approx, voltage_to_approx, '-o')
    n = 10
    polycoeff = np.polyfit(voltage_time_to_approx, voltage_to_approx, n)
    print(polycoeff)
    poly_line_array = []
    for i in range(n-1):
        line = f'${polycoeff[i]}t^{n-i}'
        poly_line_array.append(line)
    poly_line_array.append(f'${polycoeff[-1]}$')
    poly_line = f'U[V](t[us]) = {"+".join(poly_line_array)}'
    polyfunc = np.poly1d(polycoeff)
    polyvoltage = polyfunc(voltage_time_to_approx)
    '''plt.plot(voltage_time_to_approx, polyvoltage)
    #plt.title(poly_line)
    plt.ylabel('$U_{Tektronix}, V$')
    plt.xlabel('t,us')
    plt.show()'''
    plt.plot(current_time, current_amp)
    plt.show()

    plt.plot(current_time, current_amp / current_amp.max())
    # plt.plot(sinc_time, sinc_volt / sinc_volt.max())
    smoothing_shift = 0.5 * Rogovski_conv * np.gradient(current_time).mean()
    energy = np.zeros(Power.size)
    dt = np.gradient(current_time).mean()
    for i, p in enumerate(Power[1:]):
        energy[i] = energy[i - 1] + p * dt * 1.0e-6
    plt.plot(current_time, volt_volt / volt_volt.max())
    plt.plot(current_time - smoothing_shift, U_res / U_res.max())
    plt.plot(current_time - 2 * smoothing_shift, Power / Power.max())
    plt.plot(current_time - 2 * smoothing_shift, Resistance / Resistance.max())
    # plt.plot(sinc_time[peaks], sinc_volt[peaks] / sinc_volt.max(), 'o')
    # plt.plot(peak_times, current_amp[peaks] / current_amp.max(), 'o')
    plt.show()
    ret = {
        'time': current_time,
        'time_power': current_time - 2 * smoothing_shift,
        'current': current_amp,
        'u_resistive': U_res,
        'peaks': peak_times,
        'power': Power,
        'resistance': Resistance,
        'energy': energy
    }
    pd.DataFrame({
        'Time': current_time - 2 * smoothing_shift,
        'Power': Power * 1.0e-9
    }).to_csv('Power.csv')

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
        if (fname.split('.')[-1] == 'csv') & (fname[0] == 's'):
            files_data['waveform'] = open_csv(fname, -files_data['info']['Value']['Rogovski_ampl'],
                                              files_data['info']['Value']['Rogovski_conv'],
                                              files_data['info']['Value']['Inductance'])
            continue
        '''if fname.split('.')[-1] == 'xlsx':
            files_data[fname] = open_xlsx(fname)
            continue'''
    pass
    return files_data
