import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as sk_metrics

OUT_FOLDER = None
RESULT_FILE = "results.txt"


def set_output(dir_name="out"):
    global OUT_FOLDER
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    OUT_FOLDER = dir_name


def write_out(file_name, data, mode="a"):
    if OUT_FOLDER is None:
        return
    dest = os.path.join(OUT_FOLDER, file_name)
    with open(dest, mode) as out:
        out.write(data + '\n')


def write_out_fig(name):
    if OUT_FOLDER is None:
        return
    dest = os.path.join(OUT_FOLDER, name)
    plt.savefig(dest)
    plt.close()


class MyLogisticRegression(object):
    def __init__(self, **kwargs):
        self.sample_weight = None

        # method handling
        method = kwargs.pop("solver", "grad")
        allowed_methods = ("grad", "stochastic")
        if method not in allowed_methods:
            raise ValueError(
                "Method should be among {}, while {} is provided".format(
                    allowed_methods, method
                ))
        self.method = method
        self.params = kwargs

    @staticmethod
    def logistic_func(x, w):
        return 1 / (1.0 + math.e ** -np.dot(w, x))

    @staticmethod
    def _get_grad(X, y, weight, upd_index, batch_size=None):
        grad = 0
        if batch_size is None:
            indexes = range(X.shape[0])
        else:
            indexes = np.random.randint(X.shape[0], size=batch_size)
        for i in indexes:
            grad += ((y[i] - MyLogisticRegression.logistic_func(X[i], weight))
                     * X[i][upd_index])
        return grad

    def _fit_stochastic(self, X, y):
        weight = np.repeat(1.0, X.shape[1])
        params_alpha = self.params.get('alpha', None)
        params_stop = self.params.get('stop', 0.0000001)
        params_iters = self.params.get('iters', 1000)
        batch_size = self.params.get('batch_size', 20)

        scores_for_out = []

        for it in range(params_iters):
            new_weight = weight.copy()
            alpha = 2.0 / (it + 1) if params_alpha is None else params_alpha
            for j in range(len(new_weight)):
                grad = self._get_grad(X, y, weight, j, batch_size=batch_size)
                new_weight[j] += alpha * grad
            if sum(x**2 for x in new_weight - weight) < params_stop:
                break
            weight = new_weight.copy()
            scores_for_out.append(self.score(X, y, weight))
        self.sample_weight = weight

        plt.plot(np.arange(len(scores_for_out)), scores_for_out, 'r')
        plt.xlabel('iteration')
        plt.ylabel('error')
        plt.title('Stochastic prediction error')
        write_out_fig("stochastic_prediction_error.png")

    def _fit_grad(self, X, y):
        weight = np.repeat(1.0, X.shape[1])
        params_alpha = self.params.get('alpha', None)
        params_stop = self.params.get('stop', 0.0000001)
        params_iters = self.params.get('iters', 1000)

        scores_for_out = []

        for it in range(params_iters):
            new_weight = weight.copy()
            alpha = 2.0 / (it + 1) if params_alpha is None else params_alpha
            for j in range(len(new_weight)):
                grad = self._get_grad(X, y, weight, j)
                new_weight[j] += alpha * grad
            if sum(x**2 for x in new_weight - weight) < params_stop:
                break
            weight = new_weight.copy()
            scores_for_out.append(self.score(X, y, weight))
        self.sample_weight = weight

        plt.plot(np.arange(len(scores_for_out)), scores_for_out, 'r')
        plt.xlabel('iteration')
        plt.ylabel('error')
        plt.title('Gradient prediction error')
        write_out_fig("gradient_prediction_error.png")

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("Wrong shapes of X: {} and y: {}".format(
                X.shape, y.shape))
        if self.method == "stochastic":
            self._fit_stochastic(X, y)
        elif self.method == "grad":
            self._fit_grad(X, y)

        return self.sample_weight

    def predict(self, X, sample_weight=None):
        probas = self.predict_proba(X, sample_weight)
        return np.asarray([0 if p < 0.5 else 1 for p in probas])

    def predict_proba(self, X, sample_weight=None):
        if self.sample_weight is None and sample_weight is None:
            raise ValueError("Sample weights are not fitted")
        if sample_weight is None:
            sample_weight = self.sample_weight
        if X.shape[1] != sample_weight.shape[0]:
            raise ValueError(
                "Wrong shapes of X: {} and sample weights: {}".format(
                    X.shape, sample_weight))
        return np.asarray([self.logistic_func(X[i], sample_weight)
                           for i in range(X.shape[0])])

    def score(self, X, y, sample_weight=None):
        f = self.predict(X, sample_weight)
        return float(len([x for x in f == y if x])) / len(y)


def get_xy(data, y_name=None):
    if y_name is None:
        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]
    else:
        y = data[y_name]
        X = data.drop(y_name, axis=1)
    return X.as_matrix(), y.as_matrix()


def data_report(X, y, weights=None, title=None):
    if title is None:
        title = 'markers for x0 and x1'
    colors = 'ro', 'go'
    for ind, class_val in enumerate(set(y)):
        ind_values = [i for i in range(len(X)) if y[i] == class_val]
        plt.plot(X[ind_values][:, 0], X[ind_values][:, 1], colors[ind])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title(title)

    if weights is not None:
        x0 = np.arange(-2, 2, 0.01)
        x1 = -x0 * weights[0] / weights[1]
        plt.plot(x0, x1, 'b')

    write_out_fig(title.replace(" ", "_").lower() + ".png")


def gradient_report(X_train, X_test, y_train, y_test):
    lr1 = MyLogisticRegression(solver="grad")
    lr1.fit(X_train, y_train)
    data_report(X_train, y_train, weights=lr1.sample_weight,
                title="Gradient dividing line")
    lr2 = LogisticRegression()
    lr2.fit(X_train, y_train)

    write_out(RESULT_FILE,
              "SkLearn LogisticRegr score: "
              "{}".format(lr2.score(X_test, y_test)))
    write_out(RESULT_FILE,
              "Custom gradient LogisticRegr score: "
              "{}".format(lr1.score(X_test, y_test)))
    write_out(RESULT_FILE,
              "Custom gradient LogisticRegr weights: "
              "{}".format(lr1.sample_weight))
    write_out(RESULT_FILE,
              "Custom gradient LogisticRegr confusion matrix:\n"
              "{}".format(sk_metrics.confusion_matrix(y_test, lr1.predict(X_test))))


def stochastic_report(X_train, X_test, y_train, y_test):
    lr1 = MyLogisticRegression(solver="stochastic", batch_size=1)
    lr1.fit(X_train, y_train)
    data_report(X_train, y_train, weights=lr1.sample_weight,
                title="Stochastic Dividing line")
    write_out(RESULT_FILE,
              "Custom stochastic LogisticRegr score: "
              "{}".format(lr1.score(X_test, y_test)))
    write_out(RESULT_FILE,
              "Custom stochastic LogisticRegr weights: "
              "{}".format(lr1.sample_weight))
    write_out(RESULT_FILE,
              "Custom stochastic LogisticRegr confusion matrix:\n"
              "{}".format(sk_metrics.confusion_matrix(y_test, lr1.predict(X_test))))


def main():
    set_output()
    X_train, y_train = get_xy(pd.read_csv("data_train.txt", sep='\s+'))
    X_test, y_test = get_xy(pd.read_csv("data_test.txt", sep='\s+'))

    data_report(X_train, y_train)
    gradient_report(X_train, X_test, y_train, y_test)
    stochastic_report(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
