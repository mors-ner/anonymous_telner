import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional
from keras.models import Model, Input, Sequential
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
import ipdb
import gensim
from wxconv import WXC
from keras.preprocessing.text import Tokenizer


class LSTMAttrs(object):
    EPOCHS = 8
    BATCH_SIZE = 64
    EMBEDDING = 40


EMBEDDING_FILE_PATH = '/path/to/downloaded/fasttext/embedding/file.vec'
fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE_PATH, binary=False)
embedding_matrix = fasttext_model.wv.syn0


class Dataset(object):
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        con = WXC(order='utf2wx', lang='tel')
        self.df['word_wx'] = self.df['word'].apply(lambda x: con.convert(x))
        self.num_unq_words = len(list(self.df['word'].unique()))
        self.sentences = self._get_sentences()
        self.tag_to_index = self._get_tag_to_index()
        self.index_to_tag = {idx: tag for tag, idx in self.tag_to_index.items()}

    def _get_sentences(self):
        agg = lambda group: [
            (word_wx, tag)
            for word_wx, tag in zip(group['word_wx'].values.tolist(), group['tag'].values.tolist())
        ]
        self.grouped = self.df.groupby("sentence_id").apply(agg)
        return [sentence for sentence in self.grouped]

    def _get_tag_to_index(self):
        tags = list(self.df['tag'].unique())
        tag_to_index = {
            tag: index + 1
            for index, tag in enumerate(tags)
        }

        tag_to_index["PAD"] = 0

        return tag_to_index

    def get_features(self):
        features = []
        for sentence in self.sentences:
            sentence_feature = [
                word for (word, _) in sentence
            ]
            features.append(sentence_feature)

        return features

    def get_labels(self):
        labels = []
        for sentence in self.sentences:
            sentence_labels = [
                self.tag_to_index[tag]
                for (_, tag) in sentence
            ]
            labels.append(sentence_labels)

        return labels


def preprocess_data(X, y, num_unq_tags):
    tokenizer = Tokenizer(num_words=embedding_matrix.shape[0])
    tokenizer.fit_on_texts(X)

    X = tokenizer.texts_to_sequences(X)

    max_len = max([len(feature) for feature in X])

    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=0)
    y = [to_categorical(i, num_classes=num_unq_tags) for i in y]

    return X, y


def get_trained_model(X_train, y_train, num_unq_words, num_unq_tags):
    """
    Training LSTM Classifier
    """
    max_len = len(X_train[0])

    model = Sequential()
    model.add(Embedding(
        embedding_matrix.shape[0],
        embedding_matrix.shape[1],
        weights=[embedding_matrix],
        trainable=False
    ))
    model.add(Bidirectional(LSTM(
        units=50,
        return_sequences=True,
        recurrent_dropout=0.1
    )))
    model.add(TimeDistributed(Dense(50, activation="relu")))
    crf = CRF(num_unq_tags)
    model.add(crf)

    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

    print(model.summary())


    history = model.fit(
        X_train,
        np.array(y_train),
        batch_size=LSTMAttrs.BATCH_SIZE,
        epochs=LSTMAttrs.EPOCHS,
        validation_split=0.1,
    )

    return history, model


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, dest='dataset_path', help='dataset path')
    args = parser.parse_args()

    return args


def main():
    args = parse()

    dataset_obj = Dataset(args.dataset_path)
    print("Loaded dataset")
    num_unq_words = dataset_obj.num_unq_words
    num_unq_tags = len(dataset_obj.tag_to_index.keys())

    X, y = (dataset_obj.get_features(), dataset_obj.get_labels())
    X, y = preprocess_data(X, y, num_unq_tags)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    history, model = get_trained_model(X_train, y_train, num_unq_words, num_unq_tags)

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)
    y_test_true = np.argmax(y_test, -1)

    y_test_true_unpadded = []
    y_pred_unpadded = []
    for i in range(len(y_test_true)):
        temp = np.trim_zeros(y_test_true[i])
        y_test_true_unpadded.append(temp)
        y_pred_unpadded.append(y_pred[i][:len(temp)])
    y_test_true = y_test_true_unpadded
    y_pred = y_pred_unpadded

    y_pred = [
        [dataset_obj.index_to_tag[word_idx] for word_idx in sentence]
        for sentence in y_pred
    ]
    y_test_true = [
        [dataset_obj.index_to_tag[word_idx] for word_idx in sentence]
        for sentence in y_test_true
    ]

    report_labels = list(dataset_obj.tag_to_index.keys())
    report_labels.remove("PAD")

    report = flat_classification_report(y_pred=y_pred, y_true=y_test_true, labels=report_labels)
    print("================================== SKLEARN CLASSIFICATION REPORT ======================================")
    print(report)
    print("================================== SEQEVAL CLASSIFICATION REPORT ======================================")
    print(classification_report(y_test_true, y_pred))
    print("================================== SEQEVAL F1 SCORE  ======================================")
    print(f1_score(y_test_true, y_pred))
    print("================================== SEQEVAL PRECISION  ======================================")
    print(precision_score(y_test_true, y_pred))
    print("================================== SEQEVAL RECALL  ======================================")
    print(recall_score(y_test_true, y_pred))
    print("================================== SEQEVAL ACCURACY  ======================================")
    print(accuracy_score(y_test_true, y_pred))


if __name__ == "__main__":
    main()
