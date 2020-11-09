from sklearn.model_selection import KFold

from biculp import BiCULPUsingKNNFactory as BiCULP
from evaluation import Evaluations
from culp.culp.link_predictors import CompatibilityScoreStrategy
from culp.culp.similarity_strategies import VectorSimilarityMetric
from parsers import read_scene_data

X, Y = read_scene_data()
classes = list(range(len(Y[0])))

k, strategy, similarity = (3, CompatibilityScoreStrategy, VectorSimilarityMetric.manhattan)
evaluation = Evaluations()
kf = KFold(n_splits=10, random_state=42, shuffle=True)

for train_index, test_index in kf.split(X):
    x_train, y_train = X[train_index], Y[train_index]
    x_test, y_test = X[test_index], Y[test_index]
    biculp = BiCULP.create(x_train, y_train, x_test, classes, similarity, k, 16)
    biculp.train()
    prediction = biculp.predict(strategy())
    evaluation.store(y_test, prediction)

results = evaluation.evaluate()
evaluation.print_results(results)
