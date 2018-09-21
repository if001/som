"""
c = 0.2 色分けはこれくらいでおk
"""

import random
import numpy as np
# import matplotlib.pyplot as plt
import pickle
# WEIGHT_FILE = "./weight/neuron.pickle"


class Neuron():
    def __init__(self, pos, nb_input, name=None):
        self.pos = pos
        self.nb_input = nb_input
        self.name = name
        self.w = [rand_feat() for i in range(nb_input)]


class SOM():
    def __init__(self, nb_input, nb_output_h, nb_output_w, load_flag=False):
        self.nb_input = nb_input
        self.neuron_set = []
        self.save_weight_file = "./weight/neuron_" + \
            str(nb_input) + "_" + "h-" + str(nb_output_h) + \
            "_" + "w-" + str(nb_output_w) + ".pickle"
        if load_flag:
            with open(self.save_weight_file, 'rb') as f:
                self.neuron_set = pickle.load(f)
        else:
            for v1 in range(0, nb_output_h):
                for v2 in range(0, nb_output_w):
                    self.neuron_set.append(Neuron([v1, v2], nb_input))

    def save_fig(self, file_name, title_surfix):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for neuron in self.neuron_set:
            html_color = SOM.rgb2hex(neuron.w)
            ax.scatter(neuron.pos[0], neuron.pos[1],
                       c=html_color, s=1200)
        ax.grid(True)
        # plt.show()
        plt.title("img :" + str(title_surfix))
        plt.savefig(file_name)
        print("save:", file_name)

    def train_on_batch(self, data, nb_current_loop, nb_loop):
        data_np = np.array(data)
        win_neuron = self.__calc_win_neuron(data)

        for neuron in self.neuron_set:
            # neuron.w += self.__neighborhood_func(
            #     win_neuron, neuron, nb_current_loop, nb_loop) * (data_np - neuron.w)
            neuron.w += self.__neighborhood_func2(
                win_neuron, neuron, nb_current_loop) * (data_np - neuron.w)
        return win_neuron

    def distance_win_data(self, win, data):
        dis = np.linalg.norm(np.array(win.w) - np.array(data))
        return dis

    def prediction(self, data):
        win = self.__calc_win_neuron(data)
        return win

    def save_weight(self):
        with open(self.save_weight_file, 'wb') as f:
            pickle.dump(self.neuron_set, f)
        print("save " + self.save_weight_file)

    def pos_max_min(self):
        pos_set = []
        for neuron in self.neuron_set:
            pos_set.append(neuron.pos)
        pos_set = np.array(pos_set)
        pos_set_t = pos_set.T

        axis_set = []
        for v in pos_set_t:
            axis_set.append([max(v), min(v)])
        return axis_set

    def __neighborhood_func(self, neuron1, neuron2, nb_current_loop, nb_loop):
        c = 1
        c = 0.0001

        # c = 0.2 色分けはこれくらいがよい
        alpha = 1 - (nb_current_loop / nb_loop)
        dis = np.linalg.norm(np.array(neuron1.pos) - np.array(neuron2.pos))
        return c * np.exp(- np.power(dis, 2) / np.power(alpha, 2))

    def __neighborhood_func2(self, neuron1, neuron2, nb_current_loop):
        l_0 = 1.0
        lmd = 2500
        sigma_0 = 10

        l = l_0 * np.exp(- nb_current_loop / lmd)

        sigma = sigma_0 * np.exp(- nb_current_loop / lmd)
        dis = np.linalg.norm(np.array(neuron1.pos) - np.array(neuron2.pos))
        theta = np.exp(- np.power(dis, 2) / 2 * np.power(sigma, 2))

        return l * theta

    def __calc_win_neuron(self, data):
        data_np = np.array(data)
        __res = []
        for neuron in self.neuron_set:
            __res.append(np.linalg.norm(data_np - np.array(neuron.w)))
        win_neuron_idx = __res.index(min(__res))
        return self.neuron_set[win_neuron_idx]

    @classmethod
    def rgb2hex(cls, color):
        color = color[::]
        color = np.array(color)
        for i in range(len(color)):
            if color[i] > 255:
                color[i] = 255
            if color[i] < 0:
                color[i] = 0

        html_color = '#%02X%02X%02X' % (
            int(color[0]), int(color[1]), int(color[2]))
        return html_color


def rand_color():
    return random.randint(0, 255)


def rand_feat():
    return random.uniform(-0.8, 0.8)


def main():
    som = SOM(3, 9, 9)

    nb_loop = 1000
    for nb_current_loop in range(nb_loop):
        data = [rand_color(), rand_color(), rand_color()]
        win = som.train_on_batch(data, nb_current_loop, nb_loop)
        if nb_current_loop % 100 == 0:
            surfix = str(nb_current_loop)
            # surfix = str(nb_current_loop) + "_" + \
            #     str(win.pos[0]) + "_" + str(win.pos[1])
            som.save_fig("./fig/figure_" + surfix +
                         ".png", nb_current_loop)


if __name__ == "__main__":
    main()
