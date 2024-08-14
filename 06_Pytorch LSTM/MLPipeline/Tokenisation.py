
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
class Tokenisation:

    def generate_token(self,data1):
        # The maximum number of words to be used. (most frequent)
        MAX_NB_WORDS = 2000
        # Max number of words in each Content.
        MAX_SEQUENCE_LENGTH = 600
        # This is fixed. Embedding
        EMBEDDING_DIM = 100
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(data1['content'].values)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        # Tokenizing the content
        X = tokenizer.texts_to_sequences(data1['content'].values)
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        return word_index, X