
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


def train(args):
    som = SOM(128, 50, 50, load_flag=args.load)
    nb_loop = args.epoch
    sentence_list = get_sentence_lists(args.data_set_path)
    data_opt = DataSetOpt()
    for nb_current_loop in range(nb_loop):
        print(nb_current_loop)
        data, _ = get_random_word_vec(data_opt, sentence_list)
        win = som.train_on_batch(data, nb_current_loop, nb_loop)
        print("loss:", som.distance_win_data(win, data))
        if nb_current_loop % 10 == 0:
            som.save_weight()


def main():
    parser = argparse.ArgumentParser(description='som')
    parser.add_argument('--data_set_path', '-d', type=str, required=True)
    parser.add_argument('--epoch', '-e', type=int, default=400)
    parser.add_argument('--load', '-l', type=bool, default=False)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
