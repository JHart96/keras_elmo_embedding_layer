import tensorflow as tf
import utils
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from elmo import ELMoEmbedding

MAX_SEQUENCE_LENGTH = 100

(x_train, y_train), (x_test, y_test) = utils.load_data(max_sequence_length=MAX_SEQUENCE_LENGTH)
idx2word = utils.get_idx2word()

print('Build model...')
sentence_input = Input(shape=(x_train.shape[1],), dtype=tf.int64)
sentence_embedding = ELMoEmbedding(idx2word=idx2word, output_mode="elmo", trainable=True)(sentence_input) # These two are interchangeable
#sentence_embedding = Embedding(len(idx2word), 1024, input_length=MAX_SEQUENCE_LENGTH, trainable=False)(sentence_input) # These two are interchangeable
convolution = Convolution1D(50, 3, padding='same', activation='relu')(sentence_embedding)
convolution = GlobalMaxPooling1D()(convolution)
dropout = Dropout(0.5)(convolution)
hidden = Dense(50, activation='relu')(dropout)
output = Dense(1, activation='sigmoid')(hidden)

model = Model(inputs=sentence_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=2, epochs=5, validation_data=(x_test, y_test))
