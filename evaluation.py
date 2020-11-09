import numpy
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score


class Evaluations:
    def __init__(self):
        self.storage = dict()
        self.storage['hamming'] = []
        self.storage['subset_accuracy'] = []
        self.storage['precision'] = []
        self.storage['micro_precision'] = []
        self.storage['macro_precision'] = []
        self.storage['recall'] = []
        self.storage['micro_recall'] = []
        self.storage['macro_recall'] = []
        self.storage['f1'] = []
        self.storage['micro_f1'] = []
        self.storage['macro_f1'] = []

    def store(self, labels, predictions):
        for key in self.storage:
            self.storage[key].append(getattr(self, key)(labels, predictions))

    def evaluate(self):
        results = dict()
        for key in self.storage:
            results[key] = dict()
            results[key]['mean'] = numpy.average(self.storage[key])
            results[key]['std'] = numpy.std(self.storage[key])
        return results

    @staticmethod
    def print_results(results):
        results_simplified = {
            key: f"{round(100 * value['mean'], 2)}±{round(100 * value['std'], 2)}"
            for key, value in results.items()
        }

        margin = max(len(key) for key in results_simplified)
        for key, value in results_simplified.items():
            print(f'{key}{" " * (margin - len(key))} = {value}')

    def hamming(self, x, y):
        return hamming_loss(x, y)

    def subset_accuracy(self, x, y):
        return accuracy_score(x, y)

    def precision(self, x, y):
        return precision_score(x, y, average="samples", zero_division=1)

    def micro_precision(self, x, y):
        return precision_score(x, y, average="micro", zero_division=1)

    def macro_precision(self, x, y):
        return precision_score(x, y, average="macro", zero_division=1)

    def recall(self, x, y):
        return recall_score(x, y, average="samples", zero_division=1)

    def micro_recall(self, x, y):
        return recall_score(x, y, average="micro", zero_division=1)

    def macro_recall(self, x, y):
        return recall_score(x, y, average="macro", zero_division=1)

    def f1(self, x, y):
        return f1_score(x, y, average="samples", zero_division=1)

    def micro_f1(self, x, y):
        return f1_score(x, y, average="micro", zero_division=1)

    def macro_f1(self, x, y):
        return f1_score(x, y, average="macro", zero_division=1)
