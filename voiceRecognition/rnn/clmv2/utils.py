import numpy as np
import collections

def file_to_word_ids(file_path, word_to_id):
    with open(file_path, 'r', encoding='utf-8') as f:
        text_raw = f.read()                                                                                                            
        character_list = list(filter(lambda x: len(x)>0, text_raw))  
        #word_list = f.read().replace('\n', '<eos>').split()
    return [word_to_id[word] for word in character_list if word in word_to_id]

# def build_vocabulary(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         word_list = f.read().replace('\n', '<eos>').split()
#         counter = collections.Counter(word_list)
#         count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
#         words, _ = list(zip(*count_pairs))
#         word_to_id = collections.OrderedDict(zip(words, range(len(words))))
#         id_to_word = collections.OrderedDict(zip(range(len(words)), words))
#     return word_to_id, id_to_word

def build_vocabulary(file_path):                                                                                                      
    with open(file_path, 'r', encoding='utf-8') as f:                                                                               
        text_raw = f.read()                                                                                                            
        char_list = list(filter(lambda x: len(x)>0, text_raw))                                                                         
        counter = collections.Counter(char_list)                                                                                       
        counter.pop('', None)                                                                                                          
        # first sort with count, next sort with alphabet                                                                               
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))                                                             
        chars, _ = list(zip(*count_pairs))                                                                                             
        char_to_id = collections.OrderedDict(zip(chars, range(len(chars))))                                                       
        id_to_char = collections.OrderedDict(zip(range(len(chars)), chars))                                                       
    return char_to_id, id_to_char

def label_to_string(y, id_to_char):
    string_set = []

    for i in range(y.shape[0]):
        labels = y[i]
        s = ''
        for j in labels:
            s = s + id_to_char[j] + ' '

        string_set.append(s[:-1])

    return string_set
    

