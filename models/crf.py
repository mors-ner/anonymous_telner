import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score


class Dataset(object):
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        self.sentences = self.get_sentences()

    def get_sentences(self):
        agg = lambda group: [
            (word, tag)
            for word, tag in zip(group['word'].values.tolist(), group['tag'].values.tolist())
        ]
        self.grouped = self.df.groupby("sentence_id").apply(agg)

        return [sentence for sentence in self.grouped]

    def get_features(self):
        features = []
        for sentence in self.sentences:
            sentence_feature = [
                {
                    'bias': 1.0,
                    'word': word,
                }
                for (word, _) in sentence
            ]
            features.append(sentence_feature)

        return features

    def get_labels(self):
        labels = []
        for sentence in self.sentences:
            sentence_labels = [
                tag
                for (_, tag) in sentence
            ]
            labels.append(sentence_labels)

        return labels


def get_trained_model(X_train, y_train):
    """
    Training CRF Classifier
    """
    classifier = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=1000,
        all_possible_transitions=False,
        verbose=True,
    )
    classifier.fit(X_train, y_train)

    return classifier


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, dest='dataset_path', help='dataset path')
    args = parser.parse_args()

    return args


def main():
    args = parse()

    dataset_obj = Dataset(args.dataset_path)
    X, y = (dataset_obj.get_features(), dataset_obj.get_labels())
    classes = np.unique(y).tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    classifier = get_trained_model(X_train, y_train)

    y_pred = classifier.predict(X_test)

    report = flat_classification_report(y_test, y_pred)
    print("================================== SKLEARN CLASSIFICATION REPORT ======================================")
    print(report)
    print("================================== SEQEVAL CLASSIFICATION REPORT ======================================")
    print(classification_report(y_test, y_pred))
    print("================================== SEQEVAL F1 SCORE  ======================================")
    print(f1_score(y_test, y_pred))
    print("================================== SEQEVAL PRECISION  ======================================")
    print(precision_score(y_test, y_pred))
    print("================================== SEQEVAL RECALL  ======================================")
    print(recall_score(y_test, y_pred))
    print("================================== SEQEVAL ACCURACY  ======================================")
    print(accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
