# Keras ELMo Embedding Layer

This is a Keras layer for ELMo embeddings. It is designed to be completely interchangeable with the built-in Keras embedding layer.

Unfortunately the layer only works on the Tensorflow backend since it depends on a Tensorflow Hub module (https://www.tensorflow.org/hub/modules/google/elmo/2).

You can find the original paper on ELMo embeddings here: https://arxiv.org/abs/1802.05365.

I've written a blog post with a high-level overview here: https://jordanhart.co.uk/2018/09/09/elmo-embeddings-layer-in-keras/.

## Requirements

* Keras 2.2.0
* NumPy 1.13.3
* Tensorflow 1.7.0
* Tensorflow Hub 0.1.1

## Usage

To import the module:

```
from elmo import ELMoEmbedding
```

Including the embedding in your architecture is as simple as replacing an existing embedding with this layer:
```
ELMoEmbedding(idx2word=idx2word, output_mode="default", trainable=True)
```

### Arguments

* `idx2word` - a dictionary where the keys are token ids and the values are the corresponding words.
* `output_mode` - a string, one of `"default"`, `"word_emb"`, `"lstm_outputs1"`, `"lstm_outputs2"`, and `"elmo"`.
* `trainable` - a boolean, whether or not to allow the embeddings to be trained.

### Input

A 2D tensor with shape `(batch_size, max_sequence_length)`.

### Output

* `"default"` output mode - a 2D tensor with shape `(batch_size, 1024)`.
* `"word_emb"` output mode - a 3D tensor with shape `(batch_size, max_sequence_length, 512)`.
* `"lstm_outputs1"` output mode - a 3D tensor with shape `(batch_size, max_sequence_length, 1024)`.
* `"lstm_outputs2"` output mode - a 3D tensor with shape `(batch_size, max_sequence_length, 1024)`.
* `"elmo"` output mode - a 3D tensor with shape `(batch_size, max_sequence_length, 1024)`.

## Examples

The following are modified examples taken from the examples directory in the Keras repository (https://github.com/keras-team/keras). The `utils` class contains some of the preprocessing code for this dataset. This repository contains all of the code needed to run these examples.

### Sentiment analysis with sentence-level ELMo embeddings

```
import tensorflow as tf
import utils
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from elmo import ELMoEmbedding

MAX_SEQUENCE_LENGTH = 100

(x_train, y_train), (x_test, y_test) = utils.load_data(max_sequence_length=MAX_SEQUENCE_LENGTH)
idx2word = utils.get_idx2word()

sentence_input = Input(shape=(x_train.shape[1],), dtype=tf.int64)
sentence_embedding = ELMoEmbedding(idx2word=idx2word)(sentence_input) # These two are interchangeable
dropout = Dropout(0.5)(sentence_embedding)
hidden = Dense(50, activation='relu')(dropout)
output = Dense(1, activation='sigmoid')(hidden)

model = Model(inputs=sentence_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=2, epochs=5, validation_data=(x_test, y_test))
```

### Sentiment analysis with word-level ELMo embeddings

```
import tensorflow as tf
import utils
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from elmo import ELMoEmbedding

MAX_SEQUENCE_LENGTH = 100

(x_train, y_train), (x_test, y_test) = utils.load_data(max_sequence_length=MAX_SEQUENCE_LENGTH)
idx2word = utils.get_idx2word()

sentence_input = Input(shape=(x_train.shape[1],), dtype=tf.int64)
sentence_embedding = ELMoEmbedding(idx2word=idx2word, output_mode="elmo", trainable=False)(sentence_input) # These two are interchangeable
convolution = Convolution1D(50, 3, padding='same', activation='relu')(sentence_embedding)
convolution = GlobalMaxPooling1D()(convolution)
dropout = Dropout(0.5)(convolution)
hidden = Dense(50, activation='relu')(dropout)
output = Dense(1, activation='sigmoid')(hidden)

model = Model(inputs=sentence_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=2, epochs=5, validation_data=(x_test, y_test))
```