
from som_mod import SOM
from data_set_opt import DataSetOpt

import matplotlib.pyplot as plt
import random
import numpy as np
from itertools import chain
import argparse


def get_sentence_lists(file_path):
    print("make wordlists")
    with open(file_path) as f:
        lines = f.read().split("\n")
    word_lists = []
    for line in lines:
        splited_line = line.split(" ")
        while('' in splited_line):
            splited_line.remove('')
        word_lists.append(splited_line)
    print("wordlist lines num :", len(word_lists))
    return word_lists[:-1]


def get_random_word_vec(data_opt, sentence_list):
    rand1 = random.randint(0, len(sentence_list) - 1)
    sentence = sentence_list[rand1]
    rand2 = random.randint(0, len(sentence) - 1)
    word = sentence[rand2]
    word_vec = data_opt.sentence_to_vec([word])
    word_vec = np.array(word_vec)
    word_vec = word_vec.reshape(128)
    return word_vec, word


class ClasterSet():
    def __init__(self):
        self.claster_list = []

    def append(self, pos, word):
        claster = self.get_claster(pos)
        if claster is None:
            self.claster_list.append(Claster(pos, word))
        else:
            claster.append(word)

    def get_claster(self, pos):
        result = None
        for claster in self.claster_list:
            if pos == claster.pos:
                result = claster
        return result


class Claster():
    def __init__(self, pos, word):
        self.pos = pos
        self.word_set = [word]

    def append(self, word):
        if word not in self.word_set:
            self.word_set.append(word)


def word_list_to_square_string(word_list):
    edge = int(np.sqrt(len(word_list))) + 1
    while(len(word_list) < edge * edge):
        word_list.append("")
    word_list = np.array(word_list)
    word_list = word_list.reshape(edge, edge)

    new_word_list = []
    for a in word_list:
        maped_a = map(str, a)
        new_word_list.append(','.join(maped_a))
    new_line = '\n'.join(new_word_list)
    return new_line


def predict(args):
    sentence_list = get_sentence_lists(args.data_set_path)
    som = SOM(128, 50, 50, load_flag=True)
    data_opt = DataSetOpt()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    claster_set = ClasterSet()
    flatten_word_list = list(chain.from_iterable(sentence_list))
    print("unique word num ", len(flatten_word_list))

    for word in flatten_word_list[:200]:
        word_vec = data_opt.sentence_to_vec([word])
        win = som.prediction(word_vec)
        ax.scatter(win.pos[0], win.pos[1], s=50)
        print(word, ":", win.pos)
        claster_set.append([win.pos[0], win.pos[1]], word)

    for claster in claster_set.claster_list:
        annotete_string = word_list_to_square_string(claster.word_set)
        ax.annotate(annotete_string, (claster.pos[0], claster.pos[1]))
    print(som.pos_max_min())
    x_lim, y_lim = som.pos_max_min()
    plt.xlim(x_lim[1], x_lim[0])
    plt.ylim(y_lim[1], y_lim[0])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='som')
    parser.add_argument('--data_set_path', '-d', type=str, required=True)
    parser.add_argument('--epoch', '-e', type=int, default=400)
    args = parser.parse_args()

    predict(args)


if __name__ == "__main__":
    main()
