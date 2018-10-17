import numpy as np
import collections

def file_to_word_ids(file_path, word_to_id):
    with open(file_path, 'r', encoding='utf-8') as f:
        word_list = f.read().replace('\n', '<eos>').split()
    return [word_to_id[word] for word in word_list if word in word_to_id]

def build_vocabulary(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        word_list = f.read().replace('\n', '<eos>').split()
        counter = collections.Counter(word_list)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        word_to_id = collections.OrderedDict(zip(words, range(len(words))))
        id_to_word = collections.OrderedDict(zip(range(len(words)), words))
    return word_to_id, id_to_word


def label_to_string(y, id_to_word):
    string_set = []

    for i in range(y.shape[0]):
        labels = y[i]
        s = ''
        for j in labels:
            s = s + id_to_word[j] + ' '

        string_set.append(s[:-1])

    return string_set
    

