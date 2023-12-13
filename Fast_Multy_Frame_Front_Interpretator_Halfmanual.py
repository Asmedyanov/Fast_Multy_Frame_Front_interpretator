"""
The class which includes the main data processing
"""
import numpy as np

# from my_math import *
from my_os import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft2, ifft2
from scipy.ndimage.filters import maximum_filter
import cv2
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

from ApproxFunc import *


class Fast_Multy_Frame_Front_Interpretator_Halfmanual:
    """
        The class which includes the main data processing
    """

    def __init__(self, *args, **kwargs):
        """The class which includes the main data processing"""
        self.data_dict = open_folder()
        self.sort_data_dict()
        self.image_preprocessing()
        self.current_action_integral_processing()

    def image_preprocessing(self):
        os.makedirs('common', exist_ok=True)
        # common preprocessing
        self.save_all_images('common/0.original.png')
        self.shape = self.before_array.shape
        self.framecount, self.frameheight, self.framewidth = self.shape
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

    def current_action_integral_processing(self):
        # separated processing by quarts
        action_integral_list = []
        for i in [0, 1, 2, 3]:
            try:
                df = self.current_action_integral_quart(i)
                action_integral_list.append(df)
            except Exception as ex:
                print(ex)
        for df in action_integral_list:
            plt.plot(df['j_10e8_A/cm^2'], df['h_10e9_A^2*s/cm^4'], '-o', label=f'quart {i}')
        action_integral_df = pd.concat(action_integral_list)
        action_integral_df.to_csv('common/6.action_integral.csv')
        plt.legend()
        plt.grid()
        plt.title('Action integral')
        plt.xlabel('$j, 10^{8} A/cm^2$')
        plt.ylabel('$h, 10^{9} A^{2}s/cm^4$')
        plt.savefig('common/6.action_integral.png')
        plt.show()

    def current_action_integral_quart(self, quart_index):
        quart_image_array_before, quart_image_array_shot = self.quart_flip(quart_index)
        self.tilt_before_list = []
        self.shift_before_list = []
        self.popt_front_1_list = []
        self.popt_front_2_list = []
        self.fronts_list = []
        self.before_front_list = []
        self.compare_approximation_list = []
        self.expansion_list = []
        for frame_index in range(self.framecount):
            counter_line = f'Quart {quart_index + 1} Frame {frame_index + 1}'
            left_border_x, left_border_y, right_border_x, right_border_y = self.quart_borders_dialog(
                quart_image_array_before[frame_index],
                dialog_name=f'{counter_line} left&right')
            image_before = quart_image_array_before[frame_index, :, left_border_x:right_border_x]
            image_shot = quart_image_array_shot[frame_index, :, left_border_x:right_border_x]
            tilt_before, shift_before = self.recognize_front_dialog(image_before, title=f'{counter_line} before line')
            self.tilt_before_list.append(tilt_before)
            self.shift_before_list.append(shift_before)

            popt_1 = self.recognize_front_dialog(image_shot, mode='shot', title=f'{counter_line} front 1')
            popt_2 = self.recognize_front_dialog(image_shot, mode='shot', title=f'{counter_line} front 2')
            # self.popt_front_1_list.append(popt_1)
            # self.popt_front_2_list.append(popt_2)
        origin_list, good_index_list = self.get_origin(self.expansion_list, self.starts)
        good_origins = np.array(origin_list)[good_index_list]
        cross_section = self.cross_section(self.study_range * self.dx)[good_index_list]
        current_list = []
        current_density_list = []
        current_action_list = []
        dt_current = np.gradient(self.wf_time).mean()
        for i, profile_index in enumerate(good_index_list):
            current = np.interp(good_origins[i], self.wf_time, self.current)
            current_list.append(current)
            current_density = current / cross_section[i] * 1.0e2  # from mm to cm
            current_density_list.append(current_density)
            args_to_int = np.argwhere((self.wf_time > 0) & (self.wf_time <= good_origins[i]))

            action = np.sum(np.square(self.current[args_to_int])) * dt_current / np.square(
                cross_section[i]) * 1.0e-2  # from A^2*us*mm^-4 to A^2*s*cm^-4
            current_action_list.append(action)

        ret_df = pd.DataFrame({
            'j_10e8_A/cm^2': np.array(current_density_list) * 1.0e-8,
            'h_10e9_A^2*s/cm^4': np.array(current_action_list) * 1.0e-9
        })
        plt.plot(ret_df['j_10e8_A/cm^2'], ret_df['h_10e9_A^2*s/cm^4'], '-o')
        plt.show()
        return ret_df

    def get_origin(self, expansion_list, starts):
        expansion_list_length = []
        for expan in expansion_list:
            expansion_list_length.append(expan.size)
        self.study_range = np.arange(min(expansion_list_length))
        expansion_list_aligned = []
        for expan in expansion_list:
            expansion_list_aligned.append(expan[:min(expansion_list_length)])
        expansion_array = np.array(expansion_list_aligned).transpose()
        time = starts
        origins_list = []
        rel_err_origin_index_list = []
        for i, dep in enumerate(expansion_array):
            try:
                dep_loc = dep[np.argwhere(dep > 0)[:, 0]]
                time_loc = time[np.argwhere(dep > 0)[:, 0]]
                bounds = ([0, time[0] * 1.0e-2], [1.0e6, time[-1]])
                popt, pcov = curve_fit(f_square_line_time_reversed, dep_loc, time_loc, bounds=bounds)
                a, c = popt
                rel_err = (np.sqrt(np.abs(np.diag(pcov))) / np.abs(popt))
                rel_err = rel_err[-1] * 100
                dep_reg = np.arange(0, dep.max(), dep.max() * 1.0e-3)
                time_reg = f_square_line_time_reversed(dep_reg, a, c)
                origins_list.append(c)
                if (rel_err < 20):
                    rel_err_origin_index_list.append(i)
                if i % 10 == 0:
                    plt.plot(time_loc, dep_loc, 'o')
                    plt.plot(time_reg, dep_reg)
            except Exception as ex:
                print(ex)
        plt.show()
        return origins_list, rel_err_origin_index_list

    def recognize_front_dialog(self, image_array, mode='line', title='front'):
        if len(self.compare_approximation_list) > 2:
            self.compare_approximation_list = []
        # fig, ax = plt.subplots(2, 2)
        image_height, image_width = image_array.shape

        fig = plt.figure()
        plt.tight_layout()
        shape = (2, 3)
        ax = {
            'Raw data': plt.subplot2grid(shape=shape, loc=(0, 0), rowspan=2),
            'Profiles': plt.subplot2grid(shape=shape, loc=(0, 1)),
            'Expansion': plt.subplot2grid(shape=shape, loc=(1, 1)),
            'Approximation': plt.subplot2grid(shape=shape, loc=(0, 2), rowspan=2),
        }
        for my_key, my_ax in ax.items():
            my_ax.set_ylabel(my_key)
        fig.suptitle(title)
        self.ret = None
        ax['Raw data'].imshow(image_array)
        ax['Approximation'].imshow(image_array)

        x = np.arange(image_width, dtype=int)
        streak_tilt = (self.right_border_y - self.left_border_y) / image_width
        self.streak_shift = self.left_border_y
        y = (streak_tilt * x + self.streak_shift).astype(int)
        y = np.where(y >= image_height, image_height - 1, y)
        y = np.where(y < 0, 0, y)
        self.y = y
        y_down = y + self.w_front
        y_down = np.where(y_down >= image_height, image_height - 1, y_down)
        self.y_down = y_down
        y_up = y - self.w_front
        y_up = np.where(y_up < 0, 0, y_up)
        self.y_up = y_up
        plot_streak_center, = ax['Raw data'].plot(x, y)
        plot_streak_up, = ax['Raw data'].plot(x, y_up)
        plot_streak_down, = ax['Raw data'].plot(x, y_down)
        self.level = 0.5
        plot_level, = ax['Profiles'].plot([0, 2.0 * self.w_front],
                                          [self.level, self.level], '-or')
        profiles_list = []
        front_points_list = []
        profiles_plot_list = []
        plot_index_list = []

        def preprocess_profile(profile):
            profile = np.convolve(profile, conv_a, mode='same')
            profile -= profile[self.w_smooth:-self.w_smooth].min()

            profile /= profile.max()
            profile = np.where(profile < 0, 0, profile)
            return profile

        conv_a = np.ones(self.w_smooth) / float(self.w_smooth)
        for index in x:
            profile = np.copy(image_array[y_up[index]:y_down[index], index])
            profile = preprocess_profile(profile)
            front_coordinate = np.argwhere(profile > self.level).max() + y_up[index]
            front_points_list.append(front_coordinate)
            profiles_list.append(profile)
            if index % (image_width // 7) == 0:
                profiles_plot_list.append(ax['Profiles'].plot(profile)[0])
                plot_index_list.append(index)
        plot_front_points, = ax['Raw data'].plot(front_points_list, 'or')
        if mode == 'line':
            popt, pcov = curve_fit(f_line, x, front_points_list)
            approximation = popt[0] * x + popt[1]
            front = popt[1] - approximation
            self.ret = popt
            if len(self.before_front_list):
                l1 = approximation.size
                l2 = self.before_front_list[0].size
                l = min([l1, l2])
                xp = np.arange(l)
                p1 = np.array([xp, approximation[:l]]).transpose()
                p2 = np.array([xp, self.before_front_list[0][:l]]).transpose()
                self.homography, mask = cv2.findHomography(p2, p1, cv2.RANSAC)
            self.before_front_list.append(approximation)
        a = -1
        b = 100
        if len(self.tilt_before_list):
            a = self.tilt_before_list[-1]
            b = self.shift_before_list[-1]
        # da_s, db_s, db_v, x0, x_p, dxt
        bounds = ([a * 1.0e-4, -1, -image_height, -image_width, 0, 0],
                  [0, 0, 0, 0, image_height, image_width])

        def f_free_style_local(t, da_s, db_s, db_v, x0, x_p, dxt):
            return f_free_style(t, a, b, da_s, db_s, db_v, x0, x_p, dxt)

        if mode == 'shot':
            popt, pcov = curve_fit(f_free_style_local, x, front_points_list, bounds=bounds)
            da_s, db_s, db_v, x0, x_p, dxt = popt
            approximation = f_free_style_local(x, da_s, db_s, db_v, x0, x_p, dxt)
            self.ret = popt
            front = b - approximation
            self.fronts_list.append(front)

            expansion = self.compare_approximation_list[0] - approximation
            self.plot_expansion, = ax['Expansion'].plot(expansion)
            for expans in self.expansion_list:
                ax['Expansion'].plot(expans, '-.')
            self.expansion_list.append(expansion)

        # self.plot_expansion, = ax['Expansion'].plot(front)
        self.plot_approximation, = ax['Approximation'].plot(approximation, 'r')
        for appr in self.compare_approximation_list:
            ax['Approximation'].plot(appr, '-.')
        self.compare_approximation_list.append(approximation)

        def refresh():
            for index in x:
                profile = np.copy(image_array[self.y_up[index]:self.y_down[index], index])
                profile = preprocess_profile(profile)
                profiles_list[index] = profile
                front_coordinate = np.argwhere(profile > self.level).max() + self.y_up[index]
                front_points_list[index] = front_coordinate
            if mode == 'line':
                popt, pcov = curve_fit(f_line, x, front_points_list)
                approximation = popt[0] * x + popt[1]
                front = popt[1] - approximation

                self.ret = popt
            if mode == 'shot':
                popt, pcov = curve_fit(f_free_style_local, x, front_points_list, bounds=bounds)
                da_s, db_s, db_v, x0, x_p, dxt = popt
                approximation = f_free_style_local(x, da_s, db_s, db_v, x0, x_p, dxt)
                self.ret = popt
                front = b - approximation
                expansion = self.compare_approximation_list[0] - approximation
                self.expansion_list[-1] = expansion
                self.fronts_list[-1] = front
                self.plot_expansion.set_ydata(expansion)
            self.compare_approximation_list[-1] = approximation

            self.plot_approximation.set_ydata(approximation)
            plot_streak_center.set_data(x, self.y)
            plot_streak_up.set_data(x, self.y_up)
            plot_streak_down.set_data(x, self.y_down)
            for i, plot in enumerate(profiles_plot_list):
                profile = profiles_list[plot_index_list[i]]
                plot.set_data(np.arange(profile.size), profile)
            plot_front_points.set_data(np.arange(len(front_points_list)), front_points_list)
            plot_level.set_ydata([self.level, self.level])
            plt.draw()

        def mouse_event_scroll(event):
            increment = 1 if event.button == 'up' else -1
            if event.inaxes.get_ylabel() == 'Raw data':
                self.streak_shift += increment * 10
                y = (streak_tilt * x + self.streak_shift).astype(int)
                y = np.where(y >= image_height, image_height - 1, y)
                y = np.where(y < 0, 0, y)
                self.y = y
                y_down = y + self.w_front
                y_down = np.where(y_down >= image_height, image_height - 1, y_down)
                self.y_down = y_down
                y_up = y - self.w_front
                y_up = np.where(y_up < 0, 0, y_up)
                self.y_up = y_up
            if event.inaxes.get_ylabel() == 'Profiles':
                level = self.level + increment * 1.0e-2
                if (level > 0) & (level < 1.0):
                    self.level = level
            refresh()

        self.cid = fig.canvas.mpl_connect('scroll_event', mouse_event_scroll)

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.tight_layout()
        plt.show()
        return self.ret

    def quart_borders_dialog(self, image_array, dialog_name='Choose the line along the considering front'):
        """
                the dialog to choose left and right border of considering quart image
                :param image_array:
                the image as a numpy array
                :param dialog_name:
                the line with the dialog message
                :return:
                left_border_x, left_border_y, right_border_x, right_border_y of the chosen center as integer
                """
        self.left_border_x, self.right_border_x = self.frameheight // 2, self.frameheight - 5
        self.left_border_y, self.right_border_y = self.framewidth // 2 - 5, 0
        try:
            fig = plt.figure()
            plt.imshow(image_array)
            plt.title(dialog_name)
            front_plot, = plt.plot([self.left_border_x, self.right_border_x],
                                   [self.left_border_y, self.right_border_y], '-or')

            def refresh():
                if self.left_border_x > self.right_border_x:
                    a = self.left_border_x
                    b = self.left_border_y
                    self.left_border_x = self.right_border_x
                    self.left_border_y = self.right_border_y
                    self.right_border_x = a
                    self.right_border_y = b
                front_plot.set_data([self.left_border_x, self.right_border_x],
                                    [self.left_border_y, self.right_border_y])
                plt.draw()

            def mouse_event_press(event):
                self.left_border_x, self.left_border_y = event.xdata, event.ydata
                refresh()

            def mouse_event_release(event):
                self.right_border_x, self.right_border_y = event.xdata, event.ydata
                refresh()

            self.cid_1 = fig.canvas.mpl_connect('button_press_event', mouse_event_press)
            self.cid_2 = fig.canvas.mpl_connect('button_release_event', mouse_event_release)
            plt.tight_layout()
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()
        except Exception as ex:
            print(f'quart_borders_dialog {ex}')
        return int(self.left_border_x), int(self.left_border_y), \
               int(self.right_border_x), int(self.right_border_y)

    def quart_flip(self, quart_index):
        """
        .---.---.
        |   |   |
        | 1 | 0 |
        |   |   |
        .---.---.
        |   |   |
        | 2 | 3 |
        |   |   |
        .---.---.
        :param quart_index:
        :return:
        """

        ret_before = np.copy(self.before_array)
        ret_shot = np.copy(self.shot_array)
        if quart_index in [2, 3]:
            ret_before = np.flip(ret_before, axis=1)
            ret_shot = np.flip(ret_shot, axis=1)
        if quart_index in [1, 2]:
            ret_before = np.flip(ret_before, axis=2)
            ret_shot = np.flip(ret_shot, axis=2)
        limit = self.framewidth // 2
        ret_before = ret_before[:, :limit]
        ret_shot = ret_shot[:, :limit]
        return ret_before, ret_shot

    def consider_half(self, quart_ind):
        """
        the function considers the half of each of 8 frames which includes the quart
        :param quart_ind: the number of considering quart
        """

        os.makedirs(f'quart{quart_ind}', exist_ok=True)
        limit = self.framewidth // 2
        if quart_ind in [0, 1]:
            """
            for the quarts i don't need a reflection. i just cut
            """
            self.before_half = self.before_array[:, :limit]
            self.shot_half = self.shot_array[:, :limit]
        else:
            """
            for the quarts i need to reflect and cut
            """
            self.before_half = self.before_array[:, limit:]
            self.before_half = np.flip(self.before_half, axis=1)
            self.shot_half = self.shot_array[:, limit:]
            self.shot_half = np.flip(self.shot_half, axis=1)
        self.consider_quart(quart_ind)

    def consider_quart(self, quart_ind):
        """
        The function founds action integral vs current density for the quart
        :param quart_ind:
        index of quart 0,1,2,3
        :return:
        """
        a_list = []
        b_list = []
        opt_list = []

        for i in range(self.framecount):
            '''
            consider the quart for each frame
            '''
            self.get_center_of_image(self.before_half[i], dialog_name=f'Center of quart {quart_ind} camera {i}')
            limit = self.center[0]
            if quart_ind in [0, 3]:
                self.before_quart = self.before_half[i, :, limit:]
                self.shot_quart = self.shot_half[i, :, limit:]
            else:
                self.before_quart = self.before_half[i, :, :limit]
                self.before_quart = np.flip(self.before_quart, axis=1)
                self.shot_quart = self.shot_half[i, :, :limit]
                self.shot_quart = np.flip(self.shot_quart, axis=1)
            self.get_end_of_front(self.before_quart, dialog_name=f'End of quart {quart_ind} camera {i}')
            a, b = self.get_line_regration(self.before_quart, dialog_name=f'Before quart {quart_ind} camera {i}')
            opt_1 = self.get_sq_line_regration(self.shot_quart, a, b,
                                               dialog_name=f'Front 1 quart {quart_ind} camera {i}')
            opt_2 = self.get_sq_line_regration(self.shot_quart, a, b,
                                               dialog_name=f'Front 2 quart {quart_ind} camera {i}')
            a_list.append(a)
            b_list.append(b)
            opt_list.append(opt_1)
            opt_list.append(opt_2)
        popt = np.array(opt_list)
        # study_range = np.arange(int(popt[:, 0].max()))
        study_range = np.arange(self.frameheight // 2)
        polynomes_before_list = []
        polynomes_shot_list = []
        for i in range(self.shape[0] * 2):
            polynome_before = a_list[i % 4] * study_range + b_list[i % 4]

            def my_func_(t):
                y = f_free_style(t, a_list[i % 4], b_list[i % 4], popt[i, 0], popt[i, 1], popt[i, 2], popt[i, 3],
                                 popt[i, 4], popt[i, 5])
                return y

            polynome_shot = my_func_(study_range)
            polynomes_before_list.append(polynome_before)
            polynomes_shot_list.append(polynome_shot)
        main_tilt = a_list[0]
        # main_shift = polynomes_before_list[0][0]
        for i in range(self.shape[0] * 2):
            polynomes_shot_list[i] = polynomes_before_list[i][0] - polynomes_shot_list[i]
            polynomes_shot_list[i] *= main_tilt / a_list[i % 4]
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
        time = self.starts
        for i, dep in enumerate(SSW_dep.transpose()):
            try:
                dep_loc = dep[np.argwhere(dep > 0)[:, 0]]
                time_loc = time[np.argwhere(dep > 0)[:, 0]]
                # poly_coef = np.polyfit(self.starts, dep, 1)
                bounds = ([0, time[0] * 1.0e-2], [1.0e6, time[-1]])
                popt, pcov = curve_fit(f_square_line_time_reversed, dep_loc, time_loc, bounds=bounds)
                # popt, pcov = curve_fit(f_square_line_time, time, dep_loc)
                # poly_coef = popt
                # t0, b = popt
                # t0, a, b = popt
                a, c = popt
                rel_err = (np.sqrt(np.abs(np.diag(pcov))) / np.abs(popt))
                print(c)
                print(rel_err)
                rel_err = rel_err[-1] * 100
                # poly_func = np.poly1d(poly_coef)
                # time_reg = np.arange(0, self.starts.max(), self.starts.max() / 100.0)
                dep_reg = np.arange(0, dep.max(), dep.max() * 1.0e-3)
                time_reg = f_square_line_time_reversed(dep_reg, a, c)
                # dep_reg = poly_func(time_reg)
                # dep_reg = np.where(dep_reg < 0, 0, dep_reg)
                # dep_reg = f_square_line_time_reversed(time_reg, a, b, c)
                origin = c  # -poly_coef[1] / poly_coef[0]
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

                            '''if current_density > current_density_list[-1]:
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
                    plt.plot(time, dep_loc, 'o')
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

    def get_sq_line_regration(self, image_array, a, b, dialog_name='Front 1'):
        image_process = image_array[:, :self.end[0]]
        h_image, w_image = image_process.shape
        fig, ax = plt.subplots(1, 3)
        ax[2].set_title(dialog_name)
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
        # profiles_x = x_center[self.w_smooth:-self.w_smooth]
        profiles_x = x_center
        profiles_plots_list = []
        conv_n = self.w_smooth
        conv_a = np.ones(conv_n) / float(conv_n)
        for x in profiles_x:
            '''try:
                profile = image_array[self.y_up[x]:self.y_down[x], x - self.w_smooth:x + self.w_smooth].mean(axis=1)
            except:'''
            profile = np.copy(image_array[self.y_up[x]:self.y_down[x], x])
            profile = np.convolve(profile, conv_a, mode='same')
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
        # da_s, db_s, db_v, x0, x_p, dxt
        bounds = ([a, -h_image, -h_image, -w_image, 0, 0],
                  [0, 0, 0, 0, w_image, w_image])

        def mouse_event_scroll(event):
            if event.inaxes is not None:
                increment = 1 if event.button == 'up' else -1
                if event.inaxes.get_title() == 'Raw data':
                    self.b_center += 5.0 * increment
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
                        conv_n = self.w_smooth
                        conv_a = np.ones(conv_n) / float(conv_n)
                        for i, x in enumerate(profiles_x):
                            '''try:
                                profile = image_array[self.y_up[x]:self.y_down[x],
                                          x - self.w_smooth:x + self.w_smooth].mean(axis=1)
                            except:'''
                            profile = np.copy(image_array[self.y_up[x]:self.y_down[x], x])

                            profile = np.convolve(profile, conv_a, mode='same')

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

                    def f_free_style_local(x, da_s, db_s, db_v, x0, x_p, dxt):
                        return f_free_style(x, a, b, da_s, db_s, db_v, x0, x_p, dxt)

                    popt, perr = curve_fit(f_free_style_local, front_list_x, front_list_y,
                                           bounds=bounds)
                    # t0, d, a1, power1, power2 = popt
                    da_s, db_s, db_v, x0, x_p, dxt = popt
                    self.optima = popt
                    # print(popt)
                    # print(t0)
                    # poly_y = f_bipower(x_center, t0, a, b, d, a1, power1, power2)
                    poly_y = f_free_style_local(x_center, da_s, db_s, db_v, x0, x_p, dxt)
                    poly_y = np.where(poly_y > 0, poly_y, 0)
                    poly_y = np.where(poly_y < h_image, poly_y, h_image - 1)
                    self.poly_y = poly_y
                    self.plot_front.set_data(front_list_x, front_list_y)
                    self.plot_level.set_ydata([self.front_level, self.front_level])
                    self.plot_poly.set_ydata(self.poly_y)
                    plt.draw()

        def mouse_event_front_level(event):
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

            def f_free_style_local(x, da_s, db_s, db_v, x0, x_p, dxt):
                return f_free_style(x, a, b, da_s, db_s, db_v, x0, x_p, dxt)

            popt, perr = curve_fit(f_free_style_local, front_list_x, front_list_y,
                                   bounds=bounds)
            # t0, d, a1, power1, power2 = popt
            da_s, db_s, db_v, x0, x_p, dxt = popt
            self.optima = popt
            # print(popt)
            # print(t0)
            # poly_y = f_bipower(x_center, t0, a, b, d, a1, power1, power2)
            poly_y = f_free_style_local(x_center, da_s, db_s, db_v, x0, x_p, dxt)
            poly_y = np.where(poly_y > 0, poly_y, 0)
            poly_y = np.where(poly_y < h_image, poly_y, h_image - 1)
            self.poly_y = poly_y
            self.plot_front.set_data(front_list_x, front_list_y)
            self.plot_level.set_ydata([self.front_level, self.front_level])
            self.plot_poly.set_ydata(self.poly_y)
            plt.draw()

        self.cid = fig.canvas.mpl_connect('button_press_event', mouse_event_front_level)
        self.cid1 = fig.canvas.mpl_connect('scroll_event', mouse_event_scroll)
        plt.tight_layout()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
        return self.optima

    def get_line_regration(self, image_array, dialog_name='Before'):
        image_process = image_array[:, :self.end[0]]
        h_image, w_image = image_process.shape
        fig, ax = plt.subplots(1, 3)
        ax[2].set_title(dialog_name)
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
        profiles_x = x_center
        conv_n = self.w_smooth
        conv_a = np.ones(conv_n) / float(conv_n)
        for x in profiles_x:
            '''try:
                profile = image_array[y_up[x]:y_down[x], x - self.w_smooth:x + self.w_smooth].mean(axis=1)
            except:'''
            profile = np.copy(image_array[y_up[x]:y_down[x], x])
            profile = np.convolve(profile, conv_a, mode='same')
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
        plt.tight_layout()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
        return self.poly_coef

    def get_end_of_front(self, image_array, dialog_name='End of quart'):
        fig = plt.figure()
        plt.imshow(image_array)
        plt.title(dialog_name)
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
        plt.tight_layout()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

    def get_center_of_image(self, image_array, dialog_name='Center'):
        """
        the dialog to choose the horizontal center of image
        :param image_array:
        the image as a numpy array
        :param dialog_name:
        the line with the dialog message
        :return:
        (x,y) of the chosen center as integer
        """
        fig = plt.figure()
        plt.imshow(image_array)
        plt.title(dialog_name)
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
        plt.tight_layout()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
        return self.center

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
        """
        The function distributes the experiment data dictionary
        :return:
        """
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
        self.starts = self.peak_times[::2]
        self.starts = self.starts[self.sequence]
        self.stops = self.peak_times[1::2]

    def cross_section(self, z):
        """
        The function to calculate the foil cross-section in direction of the current z
        :param z:
        distance from the butterfly waist in direction of current in mm
        :return:
        cross-section of the foil in square mm
        """
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
