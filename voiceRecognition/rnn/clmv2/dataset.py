import numpy as np
from utils import file_to_word_ids, build_vocabulary, label_to_string



class PTBDataset(object):
    def __init__(self, batch_size, sequence_length, data_path = './ptb_data/', seed = 123):
        
        np.random.seed(seed)
        self.batch_size = batch_size
        self.seq_len = sequence_length
        train_file = data_path + 'ptb.train_small.txt'
        valid_file = data_path + 'ptb.valid.txt'
        test_file = data_path + 'ptb.test.txt'

        #word->vector based on the word frequency of training data
        #ex) 'the' -> 0, <unk> -> 1, ..., 'wachter' -> 9999
        #whole word->vector dictionary is saved in ptb_data/ptb_word_to_id.txt
        self.word_to_id, self.id_to_word = build_vocabulary(train_file)
        train_data = file_to_word_ids(train_file, self.word_to_id)
        valid_data = file_to_word_ids(valid_file, self.word_to_id)
        test_data = file_to_word_ids(test_file, self.word_to_id)
        
        #make x, y
        n_chunk = int((len(train_data)-1)//self.seq_len)
        self.train_x = np.reshape(train_data[:n_chunk*self.seq_len], [n_chunk, self.seq_len])
        self.train_y = np.reshape(train_data[1:n_chunk*self.seq_len+1], [n_chunk, self.seq_len])

        n_chunk = int((len(valid_data)-1)//self.seq_len)
        self.valid_x = np.reshape(valid_data[:n_chunk*self.seq_len], [n_chunk,self.seq_len])
        self.valid_y = np.reshape(valid_data[1:n_chunk*self.seq_len+1], [n_chunk, self.seq_len])

        n_chunk = int((len(test_data)-1)//self.seq_len)
        self.test_x = np.reshape(test_data[:n_chunk*self.seq_len], [n_chunk, self.seq_len])
        self.test_y = np.reshape(test_data[1:n_chunk*self.seq_len+1], [n_chunk, self.seq_len])

        self.mode = '' #train, valid, or test
        self.counter = 0
        self.n_data = self.train_x.shape[0]
        self.n_batch = int(self.n_data // self.batch_size)
        self.data_idx_perm = np.random.permutation(self.n_data)
       

    def reset(self):
        self.counter = 0
        self.data_idx_perm = np.random.permutation(self.n_data)
        self.n_batch = int(self.n_data // self.batch_size)

    def set_mode(self, mode):
        self.mode = mode

        if self.mode == 'train':
            self.n_data = self.train_x.shape[0]
        elif self.mode =='valid':
            self.n_data = self.valid_x.shape[0]
        elif self.mode == 'test':
            self.n_data = self.test_x.shape[0]
        else:
            print('wrong mode')
            raise NotImplementedError

        self.reset()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.reset()

    def iter_flag(self):
        if self.counter < self.n_batch:
            return True
        else:
            return False


    def get_data(self):
        cur_idx_begin = self.counter*self.batch_size
        data_idx = self.data_idx_perm[cur_idx_begin : cur_idx_begin + self.batch_size]

        if self.mode == 'train':
            batch_x = self.train_x[data_idx]
            batch_y = self.train_y[data_idx]
        elif self.mode == 'valid':
            batch_x = self.valid_x[data_idx]
            batch_y = self.valid_y[data_idx]
        elif self.mode == 'test':
            batch_x = self.test_x[data_idx]
            batch_y = self.test_y[data_idx]

        self.counter+=1

        return batch_x, batch_y


    def summary(self):
        print('--------------data summary---------------')
        print('batch size   : ', self.batch_size)
        print('sequence len : ', self.seq_len)
        print('train_x      : ', self.train_x.shape)
        print('train_y      : ', self.train_y.shape)
        print('valid_x      : ', self.valid_x.shape)
        print('valid_y      : ', self.valid_y.shape)
        print('test_x       : ', self.test_x.shape)
        print('test_y       : ', self.test_y.shape)
        print('word_list    : ', len(self.word_to_id.keys()))
        print('first 5 words')
        c = 0
        for k in self.word_to_id.keys():
            print(k, ' : ', self.word_to_id[k])
            c+=1
            if c == 5:
                break
        print('-----------------------------------------')
            
if __name__ == '__main__':
    dataset = PTBDataset(batch_size = 20, sequence_length = 30)
    dataset.summary()
    dataset.set_mode('train')
    c = 0
    while dataset.iter_flag():
        x, y = dataset.get_data()
        c+=1

    print('batch x : ', x.shape)
    print('batch y : ', y.shape)
    print('n_batch : ', c)
    x_words = label_to_string(x, dataset.id_to_word)
    y_words = label_to_string(y, dataset.id_to_word)

    print('---x---')
    print(x_words[0])
    print('---y---')
    print(y_words[0])


