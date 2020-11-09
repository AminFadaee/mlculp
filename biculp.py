import numpy

from culp.culp.abstracts import ClassScoresStrategy
from culp.culp.classifier import CULP
from culp.culp.leg import Leg
from culp.culp.similarity_strategies import KNNSimilarityEdgesStrategy, VectorSimilarityMetric


class BiLeg(Leg):
    def _find_class_edges(self):
        class_edges = []
        for i in range(self.n):
            for index, label in enumerate(self.labels[i]):
                if label == 1:
                    class_edges.append((i, self.n + self.m + 2 * index))
                else:
                    class_edges.append((i, self.n + self.m + 2 * index + 1))
        return class_edges


class BiCULP(CULP):
    def train(self):
        self.leg = BiLeg(self.n, self.m, self.c * 2, self.labels, self.similarity_edges_strategy)

    def predict(self, class_scores_strategy: ClassScoresStrategy):
        class_scores_strategy.load_leg(self.leg)
        similarities = class_scores_strategy.compute_class_scores()
        prediction = numpy.zeros((similarities.shape[0], similarities.shape[1] // 2), dtype=int)
        for j in range(similarities.shape[0]):
            for i in range(similarities.shape[1] // 2):
                prediction[j, i] = 1 if similarities[j, i * 2] > similarities[j, i * 2 + 1] else 0
        return prediction


class BiCULPUsingKNNFactory:
    @staticmethod
    def create(train_data, train_labels, test_data, classes, similarity: VectorSimilarityMetric, k, n_jobs=1):
        similarity_edges_strategy = KNNSimilarityEdgesStrategy(train_data, test_data, similarity, k, n_jobs)
        return BiCULP(
            n=len(train_data),
            m=len(test_data),
            c=len(classes),
            labels=train_labels,
            similarity_edges_strategy=similarity_edges_strategy,
        )
