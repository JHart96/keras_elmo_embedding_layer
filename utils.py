from keras.preprocessing import sequence
from keras.datasets import imdb

def load_data(max_sequence_length):
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data()
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=max_sequence_length, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test, maxlen=max_sequence_length, padding='post', truncating='post')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    return (x_train, y_train), (x_test, y_test)

def get_idx2word():
    INDEX_FROM = 3 # word index offset

    word_to_id = imdb.get_word_index()
    word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    idx2word = {value:key for key,value in word_to_id.items()}
    return idx2word