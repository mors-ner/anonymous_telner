import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score
from sklearn.metrics import classification_report as sklearn_classification_report
from sklearn.linear_model import LogisticRegression
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score


def get_features_targets(dataset_path):
    df = pd.read_csv(dataset_path)
    X = df.drop('tag', axis=1)
    y = df.tag.values

    cv = CountVectorizer()
    X = cv.fit_transform(X.word.values)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X)

    return X_train_tfidf, y


def get_trained_model(X_train, y_train):
    """
    Training Logistic Regression Classifier
    """
    classifier = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', verbose=1)
    classifier.fit(X_train, y_train)

    return classifier


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, dest='dataset_path', help='dataset path')
    args = parser.parse_args()

    return args


def main():
    args = parse()

    X, y = get_features_targets(args.dataset_path)
    classes = np.unique(y).tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    model = get_trained_model(X_train, y_train)

    y_pred = model.predict(X_test)
    print(sklearn_classification_report(y_pred=y_pred, y_true=y_test, labels=classes))

    report = flat_classification_report(y_true=y_test, y_pred=y_pred, labels=classes)
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
