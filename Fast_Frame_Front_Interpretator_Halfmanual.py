import numpy as np

# from my_math import *
from my_os import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft2, ifft2
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion


class Fast_Frame_Front_Interpretator_Halfmanual:
    def __init__(self, *args, **kwargs):
        self.dy = 3.9e-3  # 0.007092  # mm
        self.dx = 3.9e-3  # 0.028571  # mm
        self.init_tilt_list = [0, -0.02103, -0.00424, -0.03171]
        self.h_foil = 15e-3  # mm
        self.waist = 4.0  # mm
        self.w_foil = 40.0  # mm
        self.l_foil = 25.0  # mm

        self.shot_name, self.before_array, self.shot_array, self.peak_times, current_time, current_amp = open_images()
        self.starts = self.peak_times[::2]
        self.stops = self.peak_times[1::2]
        self.before_array = self.get_norm_array(self.before_array)
        self.shot_array = self.get_norm_array(self.shot_array)

        self.before_array_specter = fft2(self.before_array)
        self.shot_array_specter = fft2(self.shot_array)

        self.before_array_specter = self.clean_picks(self.before_array_specter)
        self.shot_array_specter = self.clean_picks(self.shot_array_specter)

        self.before_array = np.abs(ifft2(self.before_array_specter))
        self.shot_array = np.abs(ifft2(self.shot_array_specter))

        '''self.before_array = self.contrast(self.before_array)
        self.shot_array = self.contrast(self.shot_array)'''

        self.before_array = self.get_norm_array(self.before_array)
        self.shot_array = self.get_norm_array(self.shot_array)
        # self.show_original()

        self.before_array = np.swapaxes(self.before_array, 1, 2)
        self.shot_array = np.swapaxes(self.shot_array, 1, 2)
        # self.show_original()
        n_image, h_image, w_image = self.before_array.shape
        self.before_array = np.array(np.split(self.before_array, 2, 1))
        self.before_array[1] = np.flip(self.before_array[1], axis=1)
        self.before_array = self.before_array.reshape(2 * n_image, h_image // 2, w_image)
        self.shot_array = np.array(np.split(self.shot_array, 2, 1))
        self.shot_array[1] = np.flip(self.shot_array[1], axis=1)
        self.shot_array = self.shot_array.reshape(2 * n_image, h_image // 2, w_image)
        # self.show_original_half()
        polynomes_list = []
        self.min_len = w_image
        for i in range(n_image):
            print(f'image {i + 1} of {n_image}')
            polynomes = self.get_polynomes(self.before_array[i], self.shot_array[i])
            polynomes_list.append(polynomes)
        norm_before_x = polynomes_list[0][0][0]
        norm_before_y = -polynomes_list[0][0][1] + polynomes_list[0][0][1][0]
        norm = (polynomes_list[0][0][1][0] - polynomes_list[0][0][1][10]) * self.dy
        cross_section = self.cross_section(norm_before_x[:self.min_len] * self.dx)  # mm^2
        polynomes_list_aligned = [[norm_before_x[:self.min_len] * self.dx, norm_before_y[:self.min_len] * self.dy]]
        for polynomes in polynomes_list:
            # x_plot0 = polynomes[0][0] * self.dx
            y_plot0 = polynomes[0][1] * self.dy
            shift = y_plot0[0]
            tilt = y_plot0[10] - y_plot0[0]
            y_plot0 -= shift
            y_plot0 *= norm / tilt
            # plt.plot(x_plot0, y_plot0)
            x_plot1 = polynomes[1][0] * self.dx
            y_plot1 = polynomes[1][1] * self.dy
            y_plot1 -= shift
            y_plot1 *= norm / tilt
            polynomes_list_aligned.append(np.array([x_plot1[:self.min_len], y_plot1[:self.min_len]]))
            # plt.plot(x_plot1, y_plot1)
        polinomes_array = np.array(polynomes_list_aligned)
        SSW_dep = polinomes_array[:, 1] - polinomes_array[0, 1]
        # points_
        origins_list = []
        currents_list = []
        # cross_section_list = []
        current_density_list = []
        action_list = []
        dt_current = np.gradient(current_time).mean()
        for i, dep in enumerate(SSW_dep.transpose()):
            time = np.insert(self.starts, 0, 0)
            poly_coef = np.polyfit(time[1:], dep[1:], 1)

            poly_func = np.poly1d(poly_coef)
            time_reg = np.arange(0, time.max(), time.max() / 100.0)
            dep_reg = poly_func(time_reg)
            dep_reg = np.where(dep_reg < 0, 0, dep_reg)
            origin = -poly_coef[1] / poly_coef[0]
            current = np.interp(origin, current_time, current_amp)
            current_density = current / cross_section[i] * 1.0e2
            if (origin > 0):
                try:
                    if len(origins_list) > 0:
                        if origin < origins_list[-1]:
                            continue
                        if origin > current_time[np.argmax(current_amp)]:
                            continue

                        if current_density < current_density_list[-1]:
                            continue
                    origins_list.append(origin)
                    currents_list.append(current)
                    current_density_list.append(current_density)
                    args_to_int = np.argwhere((current_time > 0) & (current_time <= origin))
                    action = np.sum(current_amp[args_to_int] ** 2) * dt_current / cross_section[i] ** 2 * 1.0e-2
                    action_list.append(action)
                except Exception as ex:
                    print(ex)
            if i % 20 == 0:
                plt.plot(time, dep, 'o')
                plt.plot(time_reg, dep_reg)
        current_density_array = np.array(current_density_list)
        action_array = np.array(action_list)

        '''for polynome in polynomes_list_aligned:
            plt.plot(polynome[0], polynome[1])'''
        plt.show()
        plt.plot(current_time, current_amp)
        plt.plot(origins_list, currents_list, 'o')
        plt.show()
        plt.plot(current_density_array * 1.0e-8, action_array * 1.0e-9, '-o')
        plt.grid()
        plt.xlabel('$j, x10^8 A/cm^2$')
        plt.ylabel('$h, x10^9 A^2s/cm^4$')
        plt.title('Action integral')
        plt.show()

    def cross_section(self, z):
        s = self.waist + (self.w_foil - self.waist) * z / self.l_foil
        return 2 * s * self.h_foil

    def get_polynomes(self, image_before, image_shot):
        self.Choose_control_points(image_before)
        self.tilt_before_list = []
        self.shift_before_list = []
        print('image_before')
        polynomes_before = self.my_fitting(image_before, 1)
        print('image_shot')
        polynomes_shot = self.my_fitting(image_shot, 2)
        # print(polynomes_before)

        '''x0 = polynomes_before[0][0]
        y0 = polynomes_before[0][1]
        x1 = polynomes_before[1][0]
        y1 = polynomes_before[1][1]
        plt.plot(x0, y0)
        # plt.plot(x1, y1)
        x0 = polynomes_shot[0][0]
        y0 = polynomes_shot[0][1]
        x1 = polynomes_shot[1][0]
        y1 = polynomes_shot[1][1]
        plt.plot(x0, y0)'''
        # plt.plot(x1, y1)

        # plt.show()
        return [polynomes_before, polynomes_shot]

    def my_fitting(self, image_array, poly_power=1):
        array_left = image_array[:, self.start_point[0]:self.transit_point[0]]
        array_left = np.flip(array_left, axis=1)
        array_right = image_array[:, self.transit_point[0]:self.finish_point[0]]
        print('image_left')
        left_poly = self.get_profile_half_manual(array_left, self.transit_point[1],
                                                 self.start_point[1], poly_power)
        '''print('image_right')
        right_poly = self.get_profile_half_manual(array_right, self.transit_point[1],
                                                  self.finish_point[1], poly_power)'''
        return left_poly#right_poly

    def get_profile_half_manual(self, image_array, y_0, y_1, poly_power=1):
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image_array)
        ax[2].imshow(image_array)
        w_image = image_array.shape[1]
        h_image = image_array.shape[0]

        w_front = 150
        x_center = np.arange(w_image, dtype='int')
        b = y_0
        a = (y_1 - b) / w_image
        y_center = a * x_center + b
        y_center = y_center.astype(int)
        ax[0].plot(x_center, y_center)
        y_up = y_center - w_front
        y_up = np.where(y_up < 0, 0, y_up)
        y_down = y_center + w_front
        y_down = np.where(y_down > h_image, h_image, y_down)
        ax[0].plot(x_center, y_up)
        ax[0].plot(x_center, y_down)
        w_smooth = 15
        profiles_values = []
        profiles_x = x_center[w_smooth:-w_smooth]
        for x in profiles_x:
            profile = image_array[y_up[x]:y_down[x], x - w_smooth:x + w_smooth].mean(axis=1)
            # profile += image_array[y_up[x]-1:y_down[x]-1, x - w_smooth:x + w_smooth].mean(axis=1)
            # profile += image_array[y_up[x]+1:y_down[x]+1, x - w_smooth:x + w_smooth].mean(axis=1)
            profile -= profile.min()
            profile /= profile.mean()
            if x % 100 == 5:
                # image_array[y_up[x]:y_down[x], x] = 0
                ax[1].plot(profile)
            profiles_values.append(profile)
        plt.draw()
        self.poly_y = y_center

        def mouse_event_front_level(event):
            try:
                self.plot_front.remove()
                self.plot_poly.remove()
                self.plot_level.remove()
            except Exception as ex:
                pass
            x, y = event.xdata, event.ydata
            front_level = y
            front_list_x = []
            front_list_y = []
            for i in range(len(profiles_values)):
                try:
                    front_y = np.argwhere(profiles_values[i] > front_level).max()
                    front_y += y_up[profiles_x[i]]
                    front_x = profiles_x[i]
                    front_list_x.append(front_x)
                    front_list_y.append(front_y)
                except Exception as ex:
                    pass  # print(ex)
            if poly_power == 1:
                poly_coef = np.polyfit(front_list_x, front_list_y, 1)
                poly_func = np.poly1d(poly_coef)
                self.tilt_before = poly_coef[0]
                self.shift_before = poly_coef[1]
                self.poly_y = poly_func(x_center)
            else:
                tilt_before = self.tilt_before_list[0]
                shift_before = self.shift_before_list[0]

                def f_bi_sq(t, t0):
                    al = tilt_before / (2.0 * t0)
                    cl = tilt_before * 1.0
                    dl = shift_before * 1.0
                    bl = dl + t0 * cl / 2.0
                    yl = np.where(t > t0, cl * t + dl, al * t ** 2.0 + bl)
                    return yl

                popt, perr = curve_fit(f_bi_sq, front_list_x, front_list_y,
                                       bounds=([1, ], [w_image * 0.75, ]))
                t0 = popt
                print(t0)
                self.poly_y = f_bi_sq(x_center, t0)

            self.plot_front, = ax[0].plot(front_list_x, front_list_y, 'or')
            self.plot_level, = ax[1].plot([0, 2 * w_front], [front_level, front_level], '-or')
            self.plot_poly, = ax[2].plot(x_center, self.poly_y, 'r')
            plt.draw()

        self.cid = fig.canvas.mpl_connect('button_press_event', mouse_event_front_level)
        # ax[0].imshow(image_array)
        plt.show()
        if poly_power == 1:
            self.tilt_before_list.append(self.tilt_before)
            print(f'image_tilt = {np.arctan(self.tilt_before)} rad')
            self.shift_before_list.append(self.shift_before)
        else:
            self.tilt_before_list.pop(0)
            self.shift_before_list.pop(0)
        if x_center.size < self.min_len: self.min_len = x_center.size
        return np.array([x_center, self.poly_y])

    def contrast(self, image_array):
        ret_array = (image_array * 3) ** 2
        return ret_array

    def clean_picks(self, image_array):
        n_image, h_image, w_image = image_array.shape
        ret_array = image_array
        fft_array_abs = np.abs(image_array)
        for i in range(n_image):
            max_filter = maximum_filter(fft_array_abs[i], size=100)
            max_index = np.argwhere(fft_array_abs[i] == max_filter)
            w = 10
            for xy in max_index:
                if xy in np.array([[0, 0], [h_image - 1, 0], [0, w_image - 1], [h_image, w_image]]):
                    continue
                x0 = xy[1]
                y0 = xy[0]
                ret_array[i, y0, x0] = 0
                for r in np.arange(1, w):
                    for t in np.arange(0, 2.0 * np.pi, np.pi / int(2.0 * np.pi * r)):
                        x = int(x0 + r * np.cos(t))
                        y = int(y0 + r * np.sin(t))
                        if (x < 0): x = 0
                        if (y < 0): y = 0
                        if (y >= h_image): y = h_image - 1
                        if (x >= w_image): x = w_image - 1
                        ret_array[i, y, x] = 0
        return ret_array

    def get_approx(self, profile, order):
        def f_1(x, a, b, c):
            return a * np.abs(x - b) + c

        def f_2(x, a, b, c):
            return a * x ** 2 + b * x + c

        approx_array = []
        # print(profile[0])
        x_range = np.arange(self.start_point[0], self.finish_point[0])
        for i in range(len(profile[0])):
            xdata = profile[0][i]
            ydata = profile[1][i]
            # print(xdata)
            if order == 1:
                popt, pcov = curve_fit(f_1, xdata, ydata, bounds=(
                    [-10.0, self.transit_point[0] * 0.8, 0], [10.0, self.transit_point[0] * 1.2, 2000]))
                a, b, c = popt
                approx_array.append([x_range, f_1(x_range, a, b, c)])
            if order == 2:
                popt, pcov = curve_fit(f_2, xdata, ydata)
                a, b, c = popt
                approx_array.append([x_range, f_2(x_range, a, b, c)])

        return approx_array

    def get_pofile(self, image_array):
        n_image, h_image, w_image = image_array.shape
        n_streak = 30
        w_streak = 150
        x_array = []
        y_array = []
        for i in range(n_image):
            plt.clf()
            x_temp = []
            y_temp = []
            x = self.start_point[0]
            y_center = self.start_point[1]
            a_left = (self.start_point[1] - self.transit_point[1]) / (self.start_point[0] - self.transit_point[0])
            b_left = - a_left * self.start_point[0] + self.start_point[1]
            a_right = (self.transit_point[1] - self.finish_point[1]) / (self.transit_point[0] - self.finish_point[0])
            b_right = - a_right * self.finish_point[0] + self.finish_point[1]
            while x < self.finish_point[0]:
                try:
                    streak = image_array[i, :, x:x + n_streak]
                    profile = streak.mean(axis=1)
                    y_up = y_center - w_streak
                    if y_up < 0: y_up = 0
                    y_down = y_center + w_streak
                    if y_down > self.transit_point[1]: y_down = self.transit_point[1]
                    profile = profile[y_up:y_down]
                    y = np.argwhere(profile > 1.26 * profile.mean()).max()
                    x_temp.append(x + n_streak // 2)
                    y_temp.append(y + y_up)

                except Exception as ex:
                    print(ex)
                x += n_streak
                if x < self.transit_point[0]:
                    y_center = int(x * a_left + b_left)
                else:
                    y_center = int(x * a_right + b_right)
            x_array.append(x_temp)
            y_array.append(y_temp)
        return [x_array, y_array]

    def show_arrays(self):
        fig, ax = plt.subplots(2, 4)
        for i in range(4):
            ax[0, i].imshow(self.before_array_up[i])
            ax[0, i].plot(self.before_array_up_profile[0][i], self.before_array_up_profile[1][i], 'or')
            ax[0, i].plot(self.before_array_up_approx[i][0], self.before_array_up_approx[i][1])
            ax[1, i].imshow(self.shot_array_up[i])
            ax[1, i].plot(self.shot_array_up_profile[0][i], self.shot_array_up_profile[1][i], 'or')
            ax[1, i].plot(self.shot_array_up_approx[i][0], self.shot_array_up_approx[i][1])
        plt.show()
        plt.clf()
        fig, ax = plt.subplots(2, 4)
        for i in range(4):
            ax[0, i].imshow(self.before_array_down[i])
            ax[0, i].plot(self.before_array_down_profile[0][i], self.before_array_down_profile[1][i], 'or')
            ax[0, i].plot(self.before_array_down_approx[i][0], self.before_array_down_approx[i][1])
            ax[1, i].imshow(self.shot_array_down[i])
            ax[1, i].plot(self.shot_array_down_profile[0][i], self.shot_array_down_profile[1][i], 'or')
            ax[1, i].plot(self.shot_array_down_approx[i][0], self.shot_array_down_approx[i][1])
        plt.show()

    def get_norm_array(self, image_array):
        ret = image_array.astype(float)
        for i in range(image_array.shape[0]):
            ret[i] -= ret[i].min()
            ret[i] /= ret[i].max()
        return ret

    def get_fftf_array(self, image_array):
        n_image, h_image, w_image = image_array.shape
        fft_array = fft2(image_array)
        fft_array_abs = np.abs(fft_array)
        fft_array[:, 40:100, 0:25] = 0
        fft_array[:, 740:765, 1335:1360] = 0

        # plt.imshow(np.log(fft_array_abs[0]))
        # plt.show()

        for i in range(n_image):
            max_filter = maximum_filter(fft_array_abs[i], size=200)
            max_index = np.argwhere(fft_array_abs[i] == max_filter)
            w = 30
            for xy in max_index:
                if xy in np.array([[0, 0], [h_image - 1, 0], [0, w_image - 1], [h_image, w_image]]):
                    continue
                x0 = xy[1]
                y0 = xy[0]
                fft_array[i, y0, x0] *= 0.001
                for r in np.arange(1, w):
                    for t in np.arange(0, 2.0 * np.pi, np.pi / int(2.0 * np.pi * r)):
                        x = int(x0 + r * np.cos(t))
                        y = int(y0 + r * np.sin(t))
                        if (x < 0): x = 0
                        if (y < 0): y = 0
                        if (y >= h_image): y = h_image - 1
                        if (x >= w_image): x = w_image - 1
                        fft_array[i, y, x] *= 0.001
        # fft_array_abs = np.abs(fft_array)
        ifft_array = np.abs(ifft2(fft_array))
        # plt.imshow(np.log(fft_array_abs[0])*(np.log(fft_array_abs[0])>0))
        # plt.imshow(ifft_array[0])
        # plt.show()
        return ifft_array

    def Find_optimal_level(self):
        self.line_front_x = np.arange(int(self.start_point[0]), int(self.finish_point[0]), dtype=int)
        x_min = int(self.start_point[0])
        y_min = int(self.start_point[1])
        self.array_processed[:int(y_min)] = 0
        x_max = int(self.finish_point[0])
        y_max = int(self.finish_point[1])
        x_transit = self.transit_point[0]
        a = (y_max - y_min) / (x_max - x_min)
        b = y_min - a * x_min
        self.line_front_y = (a * self.line_front_x + b).astype(int)
        self.window = 12
        self.line_front_y_min = self.line_front_y - self.window
        self.line_front_y_max = self.line_front_y + self.window
        self.front_streak = np.zeros((2 * self.window, len(self.line_front_y)))
        for i, infinum, suprenum in zip(np.arange(len(self.line_front_y)), self.line_front_y_min,
                                        self.line_front_y_max):
            self.front_streak[:, i] = self.array_processed[infinum:suprenum, self.line_front_x[i]]
            # self.front_streak[:, i] -= self.front_streak[:, i].min()
            self.front_streak[:, i] /= self.front_streak[:, i].max()
            self.front_streak[:, i] *= 255
        fig = plt.figure()
        gs = fig.add_gridspec(1, 3)
        axes = gs.subplots()
        axes[0].imshow(self.array_processed[y_min:y_max, :x_max])
        axes[0].set_title('I see front')
        axes[1].set_title('Choose the front level or close the window to continue')
        axes[2].imshow(self.array_processed[y_min:y_max, :x_max])
        axes[2].set_title('I approximate front')
        opt_level = 0
        opt_epsilon = 3.0
        opt_coef = []
        front_x_to_approx = np.arange(int(x_max))
        front_y_to_approx = np.ones(int(x_max)) * y_min

        def f_approx(t, c0, c1, c2):
            ret = np.where(t <= x_min, 0,
                           np.where(((t > x_min) & (t < x_transit)), c0 * (t - x_min),
                                    c0 * (x_transit - x_min) * ((t + c1) / (x_transit + c1)) ** c2))
            return ret + y_min

        for level in range(255):
            front_in_streak = self.line_front_y * 0
            for i, arr in enumerate(self.front_streak.transpose()):
                try:
                    armx = np.argwhere(arr > level)[0, 0]
                    front_in_streak[i] = armx
                except:
                    pass
            front_to_approx = front_in_streak + self.line_front_y_min
            front_y_to_approx[int(x_min):] = front_to_approx
            popt, pcov = curve_fit(f=f_approx, xdata=front_x_to_approx, ydata=front_y_to_approx)
            # c0, c1, c2 = popt
            e1, e2, e3 = np.sqrt(np.diag(pcov)) / np.abs(popt)
            eps = e1 + e2 + e3
            if eps < opt_epsilon:
                opt_level = level
                opt_coef = popt
                opt_epsilon = eps
        c0, c1, c2 = opt_coef
        print(opt_level)
        print(opt_epsilon)
        self.front_approx = f_approx(front_x_to_approx, c0, c1, c2)
        axes[2].plot(front_x_to_approx, self.front_approx - y_min)
        plt.show()
        front_time = front_x_to_approx * self.dt
        front_radius = (self.front_approx - y_min) * self.dx
        dataframe_to_save = pd.DataFrame()
        dataframe_to_save['time, us'] = front_time - self.timeshift
        dataframe_to_save['radius, mm'] = front_radius
        dataframe_to_save.to_csv('streak_front_data.csv')

    def get_ns_laser_wf(self):
        fname = filedialog.askopenfilename(title="Open ns laser waveform", filetypes=[("waveforms", ".CSV")])
        try:
            ns_laser_df = pd.read_csv(fname, names=['parameter', 'value', 'e1', 'time', 'voltage', 'e2'])
            args = np.argwhere(ns_laser_df['voltage'].values > 0.9 * ns_laser_df['voltage'].values.max())[:, 0]
            times = ns_laser_df['time'].values[args]
            ret = times[0]
        except:
            ns_laser_df = pd.read_csv(fname,
                                      names=['t1', 'v1', 't2', 'v2', 't3', 'v3', 't4', 'v4', 't5', 'v5', 't6', 'v6',
                                             't7', 'v7', 't8', 'v8', 't9', 'v9', 't10', 'v10', 't11', 'v11', 't12',
                                             'v12', 't13', 'v13', 't14', 'v14', 't15', 'v15', 't16', 'v16'], skiprows=1)
            plt.clf()
            plt.plot(ns_laser_df['t15'].values, ns_laser_df['v15'].values)
            plt.show()
            args = np.argmax(ns_laser_df['v15'].values)
            volt = ns_laser_df['v15'].values[:args - 1]
            arg = np.argmax(volt)
            times = ns_laser_df['t15'].values[arg]
            ret = times

        return ret

    # time = self.ns_laser_df

    def show_profiles(self):
        self.line_front_x = np.arange(int(self.start_point[0]), int(self.finish_point[0]), dtype=int)
        x_min = int(self.start_point[0])
        y_min = int(self.start_point[1])
        self.array_processed[:int(y_min)] = 0
        x_max = int(self.finish_point[0])
        y_max = int(self.finish_point[1])
        a = (y_max - y_min) / (x_max - x_min)
        b = y_min - a * x_min
        self.line_front_y = (a * self.line_front_x + b).astype(int)
        self.window = 20
        self.line_front_y_min = self.line_front_y - self.window
        self.line_front_y_max = self.line_front_y + self.window
        self.front_streak = np.zeros((2 * self.window, len(self.line_front_y)))
        for i, infinum, suprenum in zip(np.arange(len(self.line_front_y)), self.line_front_y_min,
                                        self.line_front_y_max):
            try:
                self.front_streak[:, i] = self.array_processed[infinum:suprenum, self.line_front_x[i]]
                # self.front_streak[:, i] -= self.front_streak[:, i].min()
                self.front_streak[:, i] /= self.front_streak[:, i].max()
                self.front_streak[:, i] *= 255
            except:
                self.front_streak[:, i] = self.front_streak[:, i - 1]
                # self.front_streak[:, i] -= self.front_streak[:, i].min()
                self.front_streak[:, i] /= self.front_streak[:, i].max()
                self.front_streak[:, i] *= 255
                pass
        fig = plt.figure()
        gs = fig.add_gridspec(1, 3)
        axes = gs.subplots()
        axes[0].imshow(self.array_processed[y_min:y_max, :x_max])
        axes[0].set_title('I see front')
        axes[1].set_title('Choose the front level or close the window to continue')
        axes[2].imshow(self.array_processed[y_min:y_max, :x_max])
        axes[2].set_title('I approximate front')
        plt.draw()

        # self.front_streak = (self.front_streak[:, 1:] + self.front_streak[:, :-1]) / 2
        # self.front_streak = (self.front_streak[:, 1:] + self.front_streak[:, :-1]) / 2

        def mouse_event_front_level(event):

            x, y = event.xdata, event.ydata
            self.front_level = y
            front_level_annotate = axes[1].annotate('front level', xy=(x, y), xytext=(x + 100, y),
                                                    arrowprops=dict(facecolor='red', shrink=0.05))
            front_in_streak = self.line_front_y * 0
            for i, arr in enumerate(self.front_streak.transpose()):
                try:
                    armx = np.argwhere(arr > self.front_level)[0, 0]
                    # armx_c = np.abs(armx - self.window)
                    # armx_c = np.argmin(armx_c)
                    front_in_streak[i] = armx
                except:
                    pass

            # plt.imshow(self.front_streak)
            self.front_to_approx = front_in_streak + self.line_front_y_min
            x_transit = self.transit_point[0]

            def f_approx(t, c0, c1, c2):
                ret = np.where(t <= x_min, 0,
                               np.where(((t > x_min) & (t < x_transit)), c0 * (t - x_min),
                                        c0 * (x_transit - x_min) * ((t + c1) / (x_transit + c1)) ** c2))
                '''ret = np.where(t <= x_min, 0,
                               np.where(((t > x_min) & (t < x_transit)), c0 * (t - x_min),
                                        c1 * np.log((t + c2) / (x_transit + c2)) + c0 * (x_transit - x_min)))'''
                return ret + y_min

            front_x_to_approx = np.arange(int(x_max))
            front_y_to_approx = np.ones(int(x_max)) * y_min
            front_y_to_approx[int(x_min):] = self.front_to_approx
            popt, pcov = curve_fit(f=f_approx, xdata=front_x_to_approx, ydata=front_y_to_approx)
            c0, c1, c2 = popt
            # if c2>1:
            # c2 = 1

            self.front_approx = f_approx(front_x_to_approx, c0, c1, c2)
            try:
                self.line_1.set_ydata(front_y_to_approx - y_min, 'r')
                self.line_2.set_ydata(self.front_approx - y_min, 'r')
                plt.draw()
            except Exception as error:
                print(error)
                self.line_1, = axes[0].plot(front_x_to_approx, front_y_to_approx - y_min, 'r')
                self.line_2, = axes[2].plot(front_x_to_approx, self.front_approx - y_min, 'r')
            self.front_in_streak = front_in_streak
            self.front_time = front_x_to_approx * self.dt
            self.front_radius = (self.front_approx - y_min) * self.dx
            plt.draw()

        for front in self.front_streak[::15]:
            axes[1].plot(front)
        self.cid = fig.canvas.mpl_connect('button_press_event', mouse_event_front_level)
        plt.show()
        dataframe_to_save = pd.DataFrame()
        dataframe_to_save['time, us'] = self.front_time - self.timeshift
        dataframe_to_save['radius, mm'] = self.front_radius
        dataframe_to_save.to_csv('streak_front_data.csv')

    def get_processed_image(self):
        self.processed_array = self.shot_array.astype(float)

        '''processed_array = processed_array[:, 1:] + processed_array[:, :-1]
        self.processed_array = processed_array[:,:, 1:] + processed_array[:,:, :-1]'''

        self.before_array = self.before_array.astype(float)
        for i in range(4):
            self.processed_array[i] -= self.processed_array[i].min()
            self.processed_array[i] /= self.processed_array[i].max()
            self.before_array[i] -= self.before_array[i].min()
            self.before_array[i] /= self.before_array[i].max()
            '''self.processed_array[i] = np.where(self.before_array[i] == 0, 0,
                                               np.abs(self.before_array[i] - self.processed_array[i]) /
                                               self.before_array[i])
            self.processed_array[i] = np.where(self.processed_array[i] > 1, 1, self.processed_array[i])'''
        fft_array_before = fft2(self.before_array)
        fft_array_before[:, 40:100, 0:25] = 0
        fft_array_before[:, 940:965, 1335:1360] = 0
        ifft_before = np.abs(ifft2(fft_array_before))
        self.before_array = ifft_before

        fft_array_shot = fft2(self.processed_array)
        fft_array_shot[:, 40:100, 0:25] = 0
        fft_array_shot[:, 940:965, 1335:1360] = 0
        # neighborhood = generate_binary_structure(2, 2)
        self.fft_array = fft_array_shot - fft_array_before
        '''for i in range(4):
            max_filter = maximum_filter(self.fft_array[i], size=10, mode='nearest')
            max_index = np.argwhere(self.fft_array[i] == max_filter)
            w = 2
            hory_minus = (max_index[:, 0] - w).astype(int)
            hory_plus = (max_index[:, 0] + w).astype(int)
            fft_array[i,hory_minus:hory_plus, max_index[:, 1]] *= 0'''
        self.ifft_array = np.abs(ifft2(self.fft_array))
        # self.processed_array = np.where(ifft_before == 0, 0, self.ifft_array / ifft_before)
        self.processed_array = self.ifft_array
        self.processed_array = np.where(self.processed_array > 1, 0, self.processed_array - 1.0)

    def show_original_half(self):
        shape = self.before_array.shape
        plt.imshow(self.before_array[0, 0])
        plt.show()
        plt.clf()
        fig, ax = plt.subplots(2 * shape[0], shape[1])
        for i in range(shape[1]):
            ax[0, i].imshow(self.before_array[0, i])
            ax[2, i].imshow(self.before_array[1, i])
            ax[1, i].imshow(self.shot_array[0, i])
            ax[3, i].imshow(self.shot_array[1, i])

        plt.show()

    def show_original(self):
        shape = self.before_array.shape
        plt.imshow(self.before_array[0])
        plt.show()
        plt.clf()
        fig, ax = plt.subplots(2, 4)
        for i in range(4):
            ax[0, i].imshow(self.before_array[i])
            ax[0, i].set_title(f'from {int(self.starts[i] * 1000)} ns to {int(self.stops[i] * 1000)} ns')
            ax[1, i].imshow(self.shot_array[i])
        plt.show()

    def show_specter(self):
        shape = self.before_array_specter.shape
        plt.imshow(np.log(np.abs(self.before_array_specter))[0, :shape[1] // 2, :shape[2] // 2])
        plt.show()
        plt.clf()
        fig, ax = plt.subplots(2, shape[0])
        for i in range(shape[0]):
            ax[0, i].imshow(np.log(np.abs(self.before_array_specter))[i])
            ax[1, i].imshow(np.log(np.abs(self.shot_array_specter))[i])
        plt.show()

    def Choose_control_points(self, image_array):
        fig = plt.figure()
        plt.imshow(image_array)
        plt.title('Choose the butterfly corner')
        shape = image_array.shape
        self.start_point = [0, 0]
        self.finish_point = [shape[0], 0]
        self.transit_point = [shape[0] // 2, shape[1] // 2]

        def mouse_event_front_finish(event):
            x, y = event.xdata, event.ydata
            self.finish_point = [int(x), int(y)]
            finish_annotate = plt.annotate('front finish', xy=(x, y), xytext=(x + 100, y - 100),
                                           arrowprops=dict(facecolor='red', shrink=0.05))
            plt.arrow(x=self.transit_point[0],
                      y=self.transit_point[1],
                      dx=self.finish_point[0] - self.transit_point[0],
                      dy=self.finish_point[1] - self.transit_point[1],
                      width=3)
            plt.draw()
            fig.canvas.mpl_disconnect(self.cid)
            plt.title('Corner is approximately here. Close the window to continue')

        def mouse_event_front_transit(event):
            x, y = event.xdata, event.ydata
            self.transit_point = [int(x), int(y)]
            start_annotate = plt.annotate('front transit', xy=(x, y), xytext=(x + 100, y - 100),
                                          arrowprops=dict(facecolor='red', shrink=0.05))
            plt.draw()
            # print('x: {} and y: {}'.format(event.xdata, event.ydata))
            plt.arrow(x=self.start_point[0],
                      y=self.start_point[1],
                      dx=self.transit_point[0] - self.start_point[0],
                      dy=self.transit_point[1] - self.start_point[1],
                      width=3)
            fig.canvas.mpl_disconnect(self.cid)
            self.cid = fig.canvas.mpl_connect('button_press_event', mouse_event_front_finish)
            plt.title('Choose the finish point')

        def mouse_event_front_start(event):
            x, y = event.xdata, event.ydata
            self.start_point = [int(x), int(y)]
            start_annotate = plt.annotate('front start', xy=(x, y), xytext=(x + 100, y - 100),
                                          arrowprops=dict(facecolor='red', shrink=0.05))
            plt.draw()
            # print('x: {} and y: {}'.format(event.xdata, event.ydata))
            fig.canvas.mpl_disconnect(self.cid)
            self.cid = fig.canvas.mpl_connect('button_press_event', mouse_event_front_transit)
            plt.title('Choose the transition point')

        self.cid = fig.canvas.mpl_connect('button_press_event', mouse_event_front_start)
        plt.show()
