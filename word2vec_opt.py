from gensim.models import word2vec
import argparse

W2V_WEIGHT_PATH = "weight/w2v.model"
WORD_FEAT_LEN = 128


class MyWord2Vec():
    def __init__(self, data_set_file_path=""):
        if data_set_file_path == "":
            self.model = word2vec.Word2Vec.load(W2V_WEIGHT_PATH)
        else:
            sentences = word2vec.Text8Corpus(data_set_file_path)
            self.model = word2vec.Word2Vec(
                sentences, size=WORD_FEAT_LEN, window=5, workers=4, min_count=1, hs=1)
            self.model.save(W2V_WEIGHT_PATH)
            print("save ", W2V_WEIGHT_PATH)

    def vec_to_word(self, vec):
        return self.model.most_similar([vec], [], 1)[0][0]

    def vec_to_some_word(self, vec, num):
        return self.model.most_similar([vec], [], num)

    def word_to_vec(self, st):
        return self.model.wv[st]


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('--datasetpath', '-d', type=str, required=True)
    args = parser.parse_args()

    w2v = MyWord2Vec(args.datasetpath)

    # res = w2v.word_to_vec("私")
    # print(res)


if __name__ == "__main__":
    main()
