import numpy

from culp.culp.abstracts import ClassScoresStrategy, SimilarityEdgesStrategy
from culp.culp.classifier import CULP
from culp.culp.leg import Leg
from culp.culp.similarity_strategies import KNNSimilarityEdgesStrategy, VectorSimilarityMetric


class MiLeg(Leg):
    def _find_class_edges(self):
        class_edges = []
        for i in range(self.n):
            for index, label in enumerate(self.labels[i]):
                if label:
                    class_edges.append((i, self.n + self.m + index))
        return class_edges


class MiCULP(CULP):
    def __init__(self, n, m, c, labels, similarity_edges_strategy: SimilarityEdgesStrategy, labels_cut_threshold):
        super().__init__(n, m, c, labels, similarity_edges_strategy)
        self.labels_cut_threshold = labels_cut_threshold

    def train(self):
        self.leg = MiLeg(self.n, self.m, self.c, self.labels, self.similarity_edges_strategy)

    def predict(self, class_scores_strategy: ClassScoresStrategy):
        class_scores_strategy.load_leg(self.leg)
        similarities = class_scores_strategy.compute_class_scores()
        normalizer = similarities.sum(axis=1).reshape(self.m, 1)
        normalizer[normalizer == 0] = 1
        similarities = numpy.divide(similarities, normalizer)
        prediction = 1 * (similarities > self.labels_cut_threshold)
        return prediction


class MiCULPUsingKNNFactory:
    @staticmethod
    def create(train_data, train_labels, test_data, classes, similarity: VectorSimilarityMetric, k, threshold,
               n_jobs=1):
        similarity_edges_strategy = KNNSimilarityEdgesStrategy(train_data, test_data, similarity, k, n_jobs)
        return MiCULP(
            n=len(train_data),
            m=len(test_data),
            c=len(classes),
            labels=train_labels,
            similarity_edges_strategy=similarity_edges_strategy,
            labels_cut_threshold=threshold
        )
