import numpy as np
from itertools import chain
import argparse
import sys
from word2vec_opt import MyWord2Vec


class DataSetOpt():
    def __init__(self, w2v_train_file=""):
        self.w2v = MyWord2Vec(w2v_train_file)

    def get_start_token(self, start="BOS"):
        return self.sentence_to_vec([start])

    def vec_to_word(self, vec):
        return self.w2v.vec_to_word(vec)

    def sentence_to_vec(self, sentence):
        sentence_vec = []
        for s in sentence:
            sentence_vec.append(self.w2v.word_to_vec(s))
        feat_len = len(sentence_vec[0])
        sentence_vec = np.array(sentence_vec)
        sentence_vec = sentence_vec.reshape(1, len(sentence), feat_len)
        return sentence_vec

    def sentence_to_vec_with_BOS(self):
        pass

    def sentence_to_vec_with_EOS(self):
        pass


def get_word_lists(file_path):
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


def reshape_and_save(data_opt, word_list, output_file):
    train = []
    teach = []
    target = []

    flatten_word_list = list(chain.from_iterable(word_list))
    print("word num:", len(flatten_word_list))
    unique_word_list = ["BOS"] + ["EOS"] + list(set(flatten_word_list))
    print("unique word num:", len(unique_word_list))
    for i in range(len(word_list) - 1):
        sys.stdout.write("\r now:(%d/%d)" % (i, len(word_list)))
        sys.stdout.flush()
        p = data_opt.sentence_to_vec(word_list[i][::-1])
        print(p)
        exit(0)
        train.append()
        target.append()
        teach.append(data_opt.sentence_to_vec(word_list[i + 1] + ["EOS"]))

    np.savez_compressed(output_file, dict=unique_word_list,
                        train=train, teach=teach, target=target)
    print("save " + output_file)


def main():
    parser = argparse.ArgumentParser(description='save')
    parser.add_argument('--src_file', '-i', type=str)
    args = parser.parse_args()

    if args.src_file is None:
        data_opt = DataSetOpt()
    else:
        data_opt = DataSetOpt(args.src_file)

    p = data_opt.sentence_to_vec(["私", "は"])
    p = np.array(p)
    print(p.shape)


if __name__ == "__main__":
    main()
