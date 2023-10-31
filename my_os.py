from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt


def open_images():
    #fname = filedialog.askopenfilename(title="Open image before", filetypes=[("images", ".rtv")])
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
    #fname = filedialog.askopenfilename(title="open image shot", filetypes=[("images", ".rtv")])
    fname = 'shot25.rtv'
    file = open(fname, 'rb')
    file_array = np.fromfile(file, dtype='uint16', offset=0x2000, count=n * 4).reshape((4, 1024, 1360))

    ar_right = np.copy(file_array[1::2, :, :1360 // 2])
    ar_left = np.copy(file_array[1::2, :, 1360 // 2:])
    file_array[1::2, :, :1360 // 2] = ar_left
    file_array[1::2, :, 1360 // 2:] = ar_right

    shot_array = np.copy(file_array)

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
    return fname,before_array, shot_array
