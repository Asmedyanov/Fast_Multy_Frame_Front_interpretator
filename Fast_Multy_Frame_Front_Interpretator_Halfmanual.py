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
import sys

from ApproxFunc import *


class Fast_Multy_Frame_Front_Interpretator_Halfmanual:
    """
        The class which includes the main data processing
    """

    def __init__(self, *args, **kwargs):
        """The class which includes the main data processing"""
        self.data_dict = open_folder()
        self.curdir = os.curdir
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
                self.quart_index = i
                df = self.current_action_integral_quart()
                action_integral_list.append(df)
            except Exception as e:
                print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        for i, df in enumerate(action_integral_list):
            plt.plot(df['j_10e8_A/cm^2'], df['h_10e9_A^2*s/cm^4'], '-o', label=f'quart {i}')
        action_integral_df = pd.concat(action_integral_list)
        action_integral_df.to_csv('common/6.action_integral.csv')
        plt.legend()
        plt.grid()
        plt.title('Action integral')
        plt.xlabel('$j, 10^8 \\times A/cm^2$')
        plt.ylabel('$h, 10^9 \\times A^{2}s/cm^4$')
        plt.savefig('common/6.action_integral.png')
        plt.show()

    def current_action_integral_quart(self):
        quart_image_array_before, quart_image_array_shot = self.quart_flip(self.quart_index)

        self.tilt_before_list = []
        self.shift_before_list = []
        self.popt_front_1_list = []
        self.popt_front_2_list = []
        self.fronts_list = []
        self.before_front_list = []
        self.compare_approximation_list = []
        self.expansion_list = []
        self.levels_list = []
        self.left_border_x_list = []
        self.left_border_y_list = []
        self.right_border_x_list = []
        self.right_border_y_list = []
        self.streak_shift_list = []

        quart_line = f'quart_{self.quart_index}'
        os.makedirs(quart_line, exist_ok=True)
        self.mode = 'hard'
        try:
            self.df_approximation_ref = pd.read_csv(f'{quart_line}/approximation_parameters.csv')
            self.df_borders_ref = pd.read_csv(f'{quart_line}/frame_borders.csv')
            self.mode = 'ref'
            print('Ref mode')
        except:
            pass

        # os.chdir(f'quart_{quart_index}')
        self.front_index = 0
        for frame_index in range(self.framecount):
            self.frame_index = frame_index
            counter_line = f'Quart {self.quart_index + 1} Frame {frame_index + 1}'
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
        df_approximation = pd.DataFrame({
            'level': self.levels_list,
            'shift': self.streak_shift_list
        })
        df_borders = pd.DataFrame({
            'left_x': self.left_border_x_list,
            'left_y': self.left_border_y_list,
            'right_x': self.right_border_x_list,
            'right_y': self.right_border_y_list
        })
        df_approximation.to_csv(f'{quart_line}/approximation_parameters.csv')
        df_borders.to_csv(f'{quart_line}/frame_borders.csv')
        origin_list, good_index_list = self.get_origin(self.expansion_list, self.starts)
        good_origins = np.array(origin_list)
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
        # report about current
        plt.plot(good_origins, np.array(current_list) * 1.0e-5, 'o')
        plt.plot(self.wf_time, self.current * 1.0e-5)
        plt.title(f'Explosion current quart {self.quart_index}')
        plt.ylabel('$I, 10^5 \\times A$')
        plt.xlabel('$t, \mu s$')
        plt.savefig(f'{quart_line}/Explosion_current_{quart_line}.png')
        plt.clf()

        # report about action integral
        ret_df = pd.DataFrame({
            'j_10e8_A/cm^2': np.array(current_density_list) * 1.0e-8,
            'h_10e9_A^2*s/cm^4': np.array(current_action_list) * 1.0e-9
        })
        plt.plot(ret_df['j_10e8_A/cm^2'], ret_df['h_10e9_A^2*s/cm^4'], '-o')
        plt.title(f'Action integral quart {self.quart_index}')
        plt.xlabel('$j, 10^8 \\times A/cm^2$')
        plt.ylabel('$h, 10^9 \\times A^{2}s/cm^4$')
        plt.savefig(f'{quart_line}/Action_integral_{quart_line}.png')
        plt.show()
        ret_df.to_csv(f'{quart_line}/action_integral.csv')
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
        rel_err_list = []
        velocity_list = []
        for i, dep in enumerate(expansion_array):
            try:
                df_dep = pd.DataFrame(
                    {
                        'time': time,
                        'dep': dep
                    }
                )

                dep_loc = df_dep['dep'].values
                time_loc = df_dep['time'].values
                if dep_loc.max() == 0:
                    continue
                arg_approx = np.argwhere(dep_loc > 0).min()
                if arg_approx > 0:
                    arg_approx -= 1
                dep_loc = dep_loc[arg_approx:]
                time_loc = time_loc[arg_approx:]
                if dep_loc.size < 3:
                    continue
                '''bounds = ([0, 0], [1.0e6, time_loc[-1]])
                if len(origins_list) > 0:
                    bounds = ([0, origins_list[-1]], [1.0e6, time_loc[-1]])'''
                popt, pcov = curve_fit(f_square_line_time_reversed, dep_loc, time_loc)
                a, c = popt
                dep_reg = np.arange(0, dep.max(), dep.max() * 1.0e-3)
                time_reg = f_square_line_time_reversed(dep_reg, a, c)
                rel_err = np.sqrt(np.square(f_square_line_time_reversed(dep_loc, a, c) / time_loc - 1).mean()) * 100
                rel_err_list.append(rel_err)

                if (rel_err < 20):
                    rel_err_origin_index_list.append(i)
                    origins_list.append(c)
                    velocity_list.append(self.dx/a)
                if (i % 20 == 0):
                    plt.plot(time_loc, dep_loc * self.dx, 'o')
                    plt.plot(time_reg, dep_reg * self.dx)

            except Exception as ex:
                print(ex)
        plt.ylabel('expansion, mm')
        plt.xlabel('$t, \mu s$')
        plt.title('Origin approximation')
        plt.savefig(f'quart_{self.quart_index}/Origin approximation.png')
        plt.show()
        plt.plot(velocity_list)
        plt.ylabel('Velosity, km/s')
        plt.xlabel('x, pix')
        plt.title('Velocity approximation')
        plt.savefig(f'quart_{self.quart_index}/Velocity approximation.png')
        plt.show()
        plt.ylabel('relative error, %')
        plt.xlabel('profile number')
        plt.title('Relative error')
        plt.plot(rel_err_list)
        plt.savefig(f'quart_{self.quart_index}/Relative_error.png')
        plt.show()
        plt.plot(origins_list)
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
            'Expansion, mm': plt.subplot2grid(shape=shape, loc=(1, 1)),
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
        self.level = 0.5
        if self.mode == 'ref':
            some_index, self.level, self.streak_shift = self.df_approximation_ref.iloc[self.front_index].values
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
        self.levels_list.append(self.level)
        self.streak_shift_list.append(self.streak_shift)
        a = -1
        b = 100
        if len(self.tilt_before_list):
            a = self.tilt_before_list[-1]
            b = self.shift_before_list[-1]
        # db_v, x0, x_p, dxt
        bounds = ([-image_height, -image_width, 0, 0],
                  [0, 0, image_height, image_width])

        def f_free_style_local(t, db_v, x0, x_p, dxt):
            return f_free_style_2(t, a, b, db_v, x0, x_p, dxt)

        if mode == 'shot':
            popt, pcov = curve_fit(f_free_style_local, x, front_points_list, bounds=bounds)
            db_v, x0, x_p, dxt = popt
            approximation = f_free_style_local(x, db_v, x0, x_p, dxt)
            '''popt, pcov = curve_fit(f_line, x, front_points_list)
            approximation = popt[0] * x + popt[1]'''
            self.ret = popt

            expansion = self.compare_approximation_list[0] - approximation
            self.plot_expansion, = ax['Expansion, mm'].plot(x * self.dx, expansion * self.dx)
            for expans in self.expansion_list:
                ax['Expansion, mm'].plot(np.arange(expans.size) * self.dx, expans * self.dx, '-.')
            self.expansion_list.append(expansion)

        # self.plot_expansion, = ax['Expansion, mm'].plot(front)
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
                db_v, x0, x_p, dxt = popt
                approximation = f_free_style_local(x, db_v, x0, x_p, dxt)
                '''popt, pcov = curve_fit(f_line, x, front_points_list)
                approximation = popt[0] * x + popt[1]'''
                self.ret = popt
                expansion = self.compare_approximation_list[0] - approximation
                self.expansion_list[-1] = expansion
                # self.fronts_list[-1] = front
                self.plot_expansion.set_data(x * self.dx, expansion * self.dx)
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
            plt.savefig(f'quart_{self.quart_index}/front_approximation.png')

            plt.draw()

        def mouse_event_scroll(event):
            increment = 1 if event.button == 'up' else -1
            if event.inaxes.get_ylabel() == 'Raw data':
                self.streak_shift += increment * 10
                self.streak_shift_list[-1] = self.streak_shift
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
                    self.levels_list[-1] = self.level
            refresh()

        self.cid = fig.canvas.mpl_connect('scroll_event', mouse_event_scroll)

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.tight_layout()
        plt.show()
        self.front_index += 1
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
        if self.mode == 'ref':
            some_index, self.left_border_x, self.left_border_y, self.right_border_x, self.right_border_y = \
                self.df_borders_ref.iloc[
                    self.frame_index].values
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
        self.left_border_x_list.append(self.left_border_x)
        self.left_border_y_list.append(self.left_border_y)
        self.right_border_x_list.append(self.right_border_x)
        self.right_border_y_list.append(self.right_border_y)

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

    def save_all_images(self, name):
        fig, ax = plt.subplots(2, 4)
        fig.set_size_inches(11.7, 8.3)
        for i in range(4):
            ax[0, i].imshow(self.before_array[i])
            ax[0, i].set_title(f'shutters {int(self.starts[::2][i] * 1000)} ns and {int(self.starts[1::2][i] * 1000)} ns')
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
        self.before_array = np.flip(self.data_dict['before'], axis=0)
        self.shot_array = np.flip(self.data_dict['shot'], axis=0)
        self.peak_times = self.data_dict['waveform']['peaks']
        self.wf_time = self.data_dict['waveform']['time']
        self.current = self.data_dict['waveform']['current']
        self.starts = self.peak_times[::2]
        self.starts = np.flip(self.starts[self.sequence])
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
