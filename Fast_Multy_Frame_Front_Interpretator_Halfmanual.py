import numpy as np

# from my_math import *
from my_os import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft2, ifft2
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

from ApproxFunc import *


class Fast_Multy_Frame_Front_Interpretator_Halfmanual:
    def __init__(self, *args, **kwargs):
        self.data_dict = open_folder()
        self.sort_data_dict()
        self.starts = self.peak_times[::2]
        self.starts = self.starts[self.sequence]
        self.stops = self.peak_times[1::2]
        os.makedirs('common', exist_ok=True)
        self.save_all_images('common/0.original.png')
        self.shape = self.before_array.shape
        self.before_array = self.get_norm_array(self.before_array)
        self.shot_array = self.get_norm_array(self.shot_array)
        self.save_all_images('common/1.normed.png')

        self.before_array_specter = fft2(self.before_array)
        self.shot_array_specter = fft2(self.shot_array)
        self.save_all_images_specter('common/2.fft2_original.png')

        self.before_array_specter = self.clean_picks(self.before_array_specter)
        self.shot_array_specter = self.clean_picks(self.shot_array_specter)
        self.save_all_images_specter('common/3.fft_cut.png')

        self.before_array = np.abs(ifft2(self.before_array_specter))
        self.shot_array = np.abs(ifft2(self.shot_array_specter))
        self.save_all_images('common/4.filtered.png')

        self.before_array = self.get_norm_array(self.before_array)
        self.shot_array = self.get_norm_array(self.shot_array)

        self.before_array = np.swapaxes(self.before_array, 1, 2)
        self.shot_array = np.swapaxes(self.shot_array, 1, 2)

        self.save_all_images('common/5.swapped.png')
        self.action_integral_list = []
        for i in [0, 1, 2, 3]:
            try:
                self.consider_quart(i)
            except Exception as ex:
                print(ex)
        plt.clf()
        for df in self.action_integral_list:
            plt.plot(df['j_10e8_A/cm^2'], df['h_10e9_A^2*s/cm^4'], '-o')
        plt.savefig('common/6.action_integral.png')
        plt.show()

    def consider_quart(self, quart_ind):
        os.makedirs(f'quart{quart_ind}', exist_ok=True)
        limit = self.shape[2] // 2
        if quart_ind in [0, 1]:
            self.before_half = self.before_array[:, :limit]
            self.shot_half = self.shot_array[:, :limit]
        else:
            self.before_half = self.before_array[:, limit:]
            self.before_half = np.flip(self.before_half, axis=1)
            self.shot_half = self.shot_array[:, limit:]
            self.shot_half = np.flip(self.shot_half, axis=1)
        self.consider_half(quart_ind)

    def consider_half(self, quart_ind):
        a_list = []
        b_list = []
        opt_list = []

        for i in range(self.shape[0]):
            self.get_center_of_image(self.before_half[i])
            limit = self.center[0]
            if quart_ind in [0, 3]:
                self.before_quart = self.before_half[i, :, limit:]
                self.shot_quart = self.shot_half[i, :, limit:]
            else:
                self.before_quart = self.before_half[i, :, :limit]
                self.before_quart = np.flip(self.before_quart, axis=1)
                self.shot_quart = self.shot_half[i, :, :limit]
                self.shot_quart = np.flip(self.shot_quart, axis=1)
            self.get_end_of_front(self.before_quart)
            a, b = self.get_line_regration(self.before_quart)
            opt = self.get_sq_line_regration(self.shot_quart, a, b)
            a_list.append(a)
            b_list.append(b)
            opt_list.append(opt)
        popt = np.array(opt_list)
        study_range = np.arange(int(popt[:, 0].max()))
        polynomes_before_list = []
        polynomes_shot_list = []
        for i in range(self.shape[0]):
            polynome_before = a_list[i] * study_range + b_list[i]

            def my_func_(t):
                # y = f_bipower(t, popt[i, 0], a_list[i], b_list[i], popt[i, 1], popt[i, 2], popt[i, 3], popt[i, 4])
                y = f_square_line(t, popt[i, 0], a_list[i], b_list[i], popt[i, 1], popt[i, 2])
                return y

            polynome_shot = my_func_(study_range)
            polynomes_before_list.append(polynome_before)
            polynomes_shot_list.append(polynome_shot)
        main_tilt = a_list[0]
        # main_shift = polynomes_before_list[0][0]
        for i in range(self.shape[0]):
            polynomes_shot_list[i] = polynomes_before_list[i][0] - polynomes_shot_list[i]
            polynomes_shot_list[i] *= main_tilt / a_list[i]
            polynomes_before_list[i] = polynomes_before_list[i][0] - polynomes_before_list[i]
        for poly in polynomes_shot_list:
            plt.plot(study_range, poly)
        plt.plot(study_range, polynomes_before_list[0])
        plt.savefig(f'quart{quart_ind}/0.profiles.png')
        plt.show()

        polynome_shot_array = np.array(polynomes_shot_list)
        SSW_dep = polynome_shot_array
        for i in range(polynome_shot_array.shape[0]):
            SSW_dep[i] -= polynomes_before_list[0]

        origins_list = []
        currents_list = []
        # cross_section_list = []
        current_density_list = []
        action_list = []
        dt_current = np.gradient(self.wf_time).mean()
        cross_section = self.cross_section(study_range * self.dx)
        time = np.insert(self.starts, 0, 0)
        for i, dep in enumerate(SSW_dep.transpose()):
            try:
                dep_loc = np.insert(dep, 0, 0)
                # poly_coef = np.polyfit(self.starts, dep, 1)
                # bounds = ([0, -100], [time[-2],100])
                # bounds = ([time[1] * 0.1, 1.0e-9, 0.2], [time[1] * 0.9, 1.0e9, 1.0])
                bounds = ([0, time[-1], -1.0e12], [time[1], time[-1] * 1.0e12, -1.0e-9])
                popt, pcov = curve_fit(f_square_line_time, time, dep_loc, bounds=bounds)
                # popt, pcov = curve_fit(f_square_line_time, time, dep_loc)
                # poly_coef = popt
                # t0, b = popt
                # t0, a, b = popt
                t0, t1, a = popt
                rel_err = (np.sqrt(np.abs(np.diag(pcov))) / np.abs(popt))
                print(t0)
                print(rel_err)
                rel_err = rel_err[0] * 100
                # poly_func = np.poly1d(poly_coef)
                time_reg = np.arange(0, self.starts.max(), self.starts.max() / 100.0)
                # dep_reg = poly_func(time_reg)
                # dep_reg = np.where(dep_reg < 0, 0, dep_reg)
                dep_reg = f_square_line_time(time_reg, t0, t1, a)
                origin = t0  # -poly_coef[1] / poly_coef[0]
                current = np.interp(origin, self.wf_time, self.current)
                current_density = current / cross_section[i] * 1.0e2
                # if (origin > np.min(self.starts) / 2.718):
                if (rel_err < 20):
                    try:
                        if len(origins_list) > 0:
                            '''if origin > origins_list[-1]:
                                continue'''
                            if origin > self.wf_time[np.argmax(self.current)]:
                                continue

                            '''if current_density < current_density_list[-1]:
                                continue'''
                        origins_list.append(origin)
                        currents_list.append(current)
                        current_density_list.append(current_density)
                        args_to_int = np.argwhere((self.wf_time > 0) & (self.wf_time <= origin))
                        action = np.sum(self.current[args_to_int] ** 2) * dt_current / cross_section[i] ** 2 * 1.0e-2
                        action_list.append(action)
                    except Exception as ex:
                        print(ex)
                if i % 10 == 0:
                    plt.plot(time, dep_loc, '-o')
                    plt.plot(time_reg, dep_reg)
                '''if ((rel_err > 30) & (origin > np.min(self.starts) / 2.718)):
                        plt.plot(self.starts, dep, '-o')
                        plt.plot(time_reg, dep_reg)'''
            except Exception as ex:
                print(ex)
        current_density_array = np.array(current_density_list)
        action_array = np.array(action_list)
        plt.savefig(f'quart{quart_ind}/1.time_reg.png')
        plt.show()
        plt.plot(self.wf_time, self.current)
        plt.plot(origins_list, currents_list, 'o')
        plt.savefig(f'quart{quart_ind}/2.current.png')
        plt.show()
        plt.plot(current_density_array * 1.0e-8, action_array * 1.0e-9, '-o')
        plt.grid()
        plt.xlabel('$j, x10^8 A/cm^2$')
        plt.ylabel('$h, x10^9 A^2s/cm^4$')
        plt.title('Action integral')
        plt.savefig(f'quart{quart_ind}/3.action_integral.png')
        plt.show()
        action_integral_data = pd.DataFrame({
            'j_10e8_A/cm^2': current_density_array * 1.0e-8,
            'h_10e9_A^2*s/cm^4': action_array * 1.0e-9
        })
        action_integral_data.to_csv(f'quart{quart_ind}/4.action_integral.csv')
        self.action_integral_list.append(action_integral_data)

    def get_sq_line_regration(self, image_array, a, b):
        image_process = image_array[:, :self.end[0]]
        h_image, w_image = image_process.shape
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image_process)
        ax[0].set_title('Raw data')
        ax[1].set_title('Level')
        ax[2].imshow(image_process)
        x_center = np.arange(w_image, dtype='int')
        b_center = self.center[1]
        self.b_center = b_center
        a_center = (self.end[1] - b_center) / w_image
        y_center = a_center * x_center + b_center
        self.y_center = y_center.astype(int)
        # ax[0].plot(x_center, y_center)
        y_up = y_center - self.w_front
        y_up = y_up.astype(int)
        self.y_up = np.where(y_up < 0, 0, y_up)
        y_down = y_center + self.w_front
        y_down = y_down.astype(int)
        self.y_down = np.where(y_down >= h_image, h_image - 1, y_down)
        self.y_up_plot, = ax[0].plot(x_center, self.y_up)
        self.y_down_plot, = ax[0].plot(x_center, self.y_down)
        self.y_center_plot, = ax[0].plot(x_center, self.y_center)
        # ax[0].plot(x_center, y_down)
        profiles_values = []
        profiles_x = x_center[self.w_smooth:-self.w_smooth]
        profiles_plots_list = []
        for x in profiles_x:
            profile = image_array[self.y_up[x]:self.y_down[x], x - self.w_smooth:x + self.w_smooth].mean(axis=1)
            profile -= profile.min()
            profile /= profile.max()
            if x % 40 == 0:
                profiles_plots_list.append(ax[1].plot(np.arange(profile.size), profile)[0])
            profiles_values.append(profile)

        self.t0_loc = 0
        self.plot_front, = ax[0].plot(x_center, self.y_center, 'or')
        self.front_level = 1.0
        self.plot_level, = ax[1].plot([0, 2 * self.w_front], [1.0, 1.0], '-or')
        self.plot_poly, = ax[2].plot(x_center, self.y_center, 'r')
        plt.draw()
        # t0, d0, t1
        # bounds = ([1, -200, 0, 0.9, 1.5, ], [w_image * 0.75, 0, 100, 1.1, 4.0, ])
        bounds = ([0, -1000, -w_image * 0.5],
                  [w_image * 0.75, 0, 0])

        def mouse_event_scroll(event):
            if event.inaxes is not None:
                increment = 1 if event.button == 'up' else -1
                if event.inaxes.get_title() == 'Raw data':
                    self.b_center += 10.0 * increment
                    new_y_center = a_center * x_center + self.b_center
                    if (np.max(new_y_center) > 0) & (np.min(new_y_center) < h_image):
                        new_y_center = np.where(new_y_center > 0, new_y_center, 0)
                        new_y_center = np.where(new_y_center < h_image, new_y_center, h_image - 1)
                        self.y_center = new_y_center
                        new_y_up = new_y_center - self.w_front
                        new_y_up = np.where(new_y_up > 0, new_y_up, 0)
                        new_y_up = np.where(new_y_up < h_image, new_y_up, h_image - 1)
                        self.y_up = new_y_up.astype(int)
                        new_y_down = new_y_center + self.w_front
                        new_y_down = np.where(new_y_down > 0, new_y_down, 0)
                        new_y_down = np.where(new_y_down < h_image, new_y_down, h_image - 1)
                        self.y_down = new_y_down.astype(int)
                        self.y_center_plot.set_ydata(self.y_center)
                        self.y_up_plot.set_ydata(self.y_up)
                        self.y_down_plot.set_ydata(self.y_down)

                        for i, x in enumerate(profiles_x):
                            profile = image_array[self.y_up[x]:self.y_down[x],
                                      x - self.w_smooth:x + self.w_smooth].mean(axis=1)
                            # profile = np.abs(np.gradient(profile))
                            profile -= profile.min()
                            profile /= profile.max()
                            if x % 40 == 0:
                                profiles_plots_list[i // 40].set_data(np.arange(profile.size), profile)
                            profiles_values[i] = profile
                        plt.draw()
                if event.inaxes.get_title() == 'Level':
                    self.front_level += 0.02 * increment
                    front_list_x = []
                    front_list_y = []
                    for i in range(len(profiles_values)):
                        try:

                            front_y = np.argwhere(profiles_values[i] > self.front_level).max()
                            front_y += self.y_up[profiles_x[i]]
                            front_x = profiles_x[i]
                            front_list_x.append(front_x)
                            front_list_y.append(front_y)
                        except Exception as ex:
                            pass  # print(ex)

                    def f_bi_power(t, t0, d, a1, power1, power2):
                        return f_bipower(t, t0, a, b, d, a1, power1, power2)

                    def f_squre_line_local(t, t0, d0, t1):
                        return f_square_line(t, t0, a, b, d0, t1)

                    popt, perr = curve_fit(f_squre_line_local, front_list_x, front_list_y,
                                           bounds=bounds)
                    # t0, d, a1, power1, power2 = popt
                    t0, d0, t1 = popt
                    self.optima = popt
                    # print(popt)
                    # print(t0)
                    # poly_y = f_bipower(x_center, t0, a, b, d, a1, power1, power2)
                    poly_y = f_squre_line_local(x_center, t0, d0, t1)
                    poly_y = np.where(poly_y > 0, poly_y, 0)
                    poly_y = np.where(poly_y < h_image, poly_y, h_image - 1)
                    self.poly_y = poly_y
                    self.plot_front.set_data(front_list_x, front_list_y)
                    self.plot_level.set_ydata([self.front_level, self.front_level])
                    self.plot_poly.set_ydata(self.poly_y)
                    plt.draw()

        def mouse_event_front_level(event):
            '''try:
                self.plot_front.remove()
                self.plot_poly.remove()
                self.plot_level.remove()
            except Exception as ex:
                pass'''
            x, y = event.xdata, event.ydata
            front_level = y
            self.front_level = front_level
            front_list_x = []
            front_list_y = []
            for i in range(len(profiles_values)):
                try:
                    front_y = np.argwhere(profiles_values[i] > front_level).max()
                    front_y += self.y_up[profiles_x[i]]
                    front_x = profiles_x[i]
                    front_list_x.append(front_x)
                    front_list_y.append(front_y)
                except Exception as ex:
                    pass  # print(ex)

            def f_bi_power(t, t0, d, a1, power1, power2):
                return f_bipower(t, t0, a, b, d, a1, power1, power2)

            def f_squre_line_local(t, t0, d0, a1):
                return f_square_line(t, t0, a, b, d0, a1)

            popt, perr = curve_fit(f_squre_line_local, front_list_x, front_list_y,
                                   bounds=bounds)
            # t0, d, a1, power1, power2 = popt
            t0, d0, a1 = popt
            self.optima = popt
            # print(popt)
            # print(t0)
            # poly_y = f_bipower(x_center, t0, a, b, d, a1, power1, power2)
            poly_y = f_squre_line_local(x_center, t0, d0, a1)
            poly_y = np.where(poly_y > 0, poly_y, 0)
            poly_y = np.where(poly_y < h_image, poly_y, h_image - 1)
            self.poly_y = poly_y
            self.plot_front.set_data(front_list_x, front_list_y)
            self.plot_level.set_ydata([self.front_level, self.front_level])
            self.plot_poly.set_ydata(self.poly_y)
            plt.draw()

        self.cid = fig.canvas.mpl_connect('button_press_event', mouse_event_front_level)
        self.cid1 = fig.canvas.mpl_connect('scroll_event', mouse_event_scroll)

        plt.show()
        return self.optima

    def get_line_regration(self, image_array):
        image_process = image_array[:, :self.end[0]]
        h_image, w_image = image_process.shape
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image_process)
        ax[2].imshow(image_process)
        x_center = np.arange(w_image, dtype='int')
        b_center = self.center[1]
        a_center = (self.end[1] - b_center) / w_image
        y_center = a_center * x_center + b_center
        y_center = y_center.astype(int)
        ax[0].plot(x_center, y_center)
        y_up = y_center - self.w_front
        y_up = np.where(y_up < 0, 0, y_up)
        y_down = y_center + self.w_front
        y_down = np.where(y_down >= h_image, h_image - 1, y_down)
        ax[0].plot(x_center, y_up)
        ax[0].plot(x_center, y_down)
        profiles_values = []
        profiles_x = x_center[self.w_smooth:-self.w_smooth]
        for x in profiles_x:
            profile = image_array[y_up[x]:y_down[x], x - self.w_smooth:x + self.w_smooth].mean(axis=1)
            profile -= profile.min()
            profile /= profile.mean()
            if x % 100 == 5:
                ax[1].plot(profile)
            profiles_values.append(profile)
        plt.draw()
        self.poly_coef = np.array([a_center, b_center])

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
            self.poly_coef = np.polyfit(front_list_x, front_list_y, 1)
            poly_func = np.poly1d(self.poly_coef)
            self.tilt_before = self.poly_coef[0]
            self.shift_before = self.poly_coef[1]
            self.poly_y = poly_func(x_center)

            self.plot_front, = ax[0].plot(front_list_x, front_list_y, 'or')
            self.plot_level, = ax[1].plot([0, 2 * self.w_front], [front_level, front_level], '-or')
            self.plot_poly, = ax[2].plot(x_center, self.poly_y, 'r')
            plt.draw()

        self.cid = fig.canvas.mpl_connect('button_press_event', mouse_event_front_level)

        plt.show()
        return self.poly_coef

    def get_end_of_front(self, image_array):
        fig = plt.figure()
        plt.imshow(image_array)
        plt.title('Choose the end of front. Close to continue')
        shape = image_array.shape
        self.end = [shape[0] - 1, shape[1] - 1]

        def mouse_event_end(event):
            try:
                self.annotate.remove()
                self.arrow.remove()
            except:
                pass
            x, y = event.xdata, event.ydata
            self.end = [int(x), int(y)]
            self.annotate = plt.annotate('front end', xy=(x, y), xytext=(x + 100, y + 100),
                                         arrowprops=dict(facecolor='red', shrink=0.05))
            self.arrow = plt.arrow(0, self.center[1], x, y - self.center[1])
            plt.draw()

        self.cid = fig.canvas.mpl_connect('button_press_event', mouse_event_end)
        plt.show()

    def get_center_of_image(self, image_array):
        fig = plt.figure()
        plt.imshow(image_array)
        plt.title('Choose the butterfly center. Close to continue')
        shape = image_array.shape
        self.center = [shape[0] // 2, shape[1] // 2]

        def mouse_event_center(event):
            try:
                self.arrow.remove()
            except:
                pass
            x, y = event.xdata, event.ydata
            self.center = [int(x), int(y)]
            self.arrow = plt.annotate('butterfly_waist', xy=(x, y), xytext=(x + 100, y - 100),
                                      arrowprops=dict(facecolor='red', shrink=0.05))
            plt.draw()

        self.cid = fig.canvas.mpl_connect('button_press_event', mouse_event_center)
        plt.show()

    def save_all_images(self, name):
        fig, ax = plt.subplots(2, 4)
        fig.set_size_inches(11.7, 8.3)
        for i in range(4):
            ax[0, i].imshow(self.before_array[i])
            ax[0, i].set_title(f'from {int(self.starts[i] * 1000)} ns to {int(self.stops[i] * 1000)} ns')
            ax[1, i].imshow(self.shot_array[i])
        plt.tight_layout()
        fig.savefig(name)
        plt.close()

    def save_all_images_specter(self, name):
        fig, ax = plt.subplots(2, 4)
        fig.set_size_inches(11.7, 8.3)
        for i in range(4):
            show_array = np.abs(self.before_array_specter[i]) + 1.0e-5
            show_array = np.where(show_array > 1.0e-3, np.log(show_array), 1.0e-3)
            ax[0, i].imshow(show_array)
            ax[0, i].set_title(f'from {int(self.starts[i] * 1000)} ns to {int(self.stops[i] * 1000)} ns')
            show_array = np.abs(self.shot_array_specter[i]) + 1.0e-5
            show_array = np.where(show_array > 1.0e-3, np.log(show_array), 1.0e-3)
            ax[1, i].imshow(show_array)
        plt.tight_layout()
        fig.savefig(name)
        plt.close()

    def sort_data_dict(self):
        self.dy = self.data_dict['info']['Value']['dy']
        self.dx = self.data_dict['info']['Value']['dx']
        self.h_foil = self.data_dict['info']['Value']['Thickness']
        self.waist = self.data_dict['info']['Value']['Waist']
        self.w_foil = self.data_dict['info']['Value']['Width']
        self.l_foil = self.data_dict['info']['Value']['Length']
        self.w_front = self.data_dict['info']['Value']['w_front']
        self.w_smooth = self.data_dict['info']['Value']['w_smooth']
        self.sequence = np.array(self.data_dict['info']['Value']['Sequence'].split(','), dtype='int')
        # self.shot_name, self.before_array, self.shot_array, self.peak_times, self.wf_time, self.current = open_images()
        self.before_array = self.data_dict['before']
        self.shot_array = self.data_dict['shot']
        self.peak_times = self.data_dict['waveform']['peaks']
        self.wf_time = self.data_dict['waveform']['time']
        self.current = self.data_dict['waveform']['current']

    def cross_section(self, z):
        s = 0.5 * self.waist + (self.w_foil - self.waist) * z / self.l_foil
        return 2 * s * self.h_foil

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
            w = 20
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
                        if (x < 0):
                            x = 0
                        if (y < 0):
                            y = 0
                        if (y >= h_image):
                            y = h_image - 1
                        if (x >= w_image):
                            x = w_image - 1
                        ret_array[i, y, x] = 0
        return ret_array

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

    def save_original_half(self, name):
        try:
            shape = self.before_array.shape
            fig, ax = plt.subplots(2 * shape[0], shape[1])
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            for i in range(shape[1]):
                ax[0, i].imshow(self.before_array[i])
                ax[2, i].imshow(self.before_array[i + shape[1] - 1])
                ax[1, i].imshow(self.shot_array[i])
                ax[3, i].imshow(self.shot_array[i + shape[1] - 1])

            fig.savefig(name)
        except Exception as ex:
            print(ex)
        plt.close()

    def show_original(self):
        plt.imshow(self.before_array[0])
        plt.show()
        plt.clf()
        fig, ax = plt.subplots(2, 4)
        for i in range(4):
            ax[0, i].imshow(self.before_array[i])
            ax[0, i].set_title(f'from {int(self.starts[i] * 1000)} ns to {int(self.stops[i] * 1000)} ns')
            ax[1, i].imshow(self.shot_array[i])
        plt.show()
