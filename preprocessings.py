from abc import abstractmethod, ABC
from typing import Union

import numpy
from sklearn.decomposition import PCA as SklearnPCA


class Preprocessor(ABC):
    @abstractmethod
    def fit(self, x: numpy.ndarray, transform: bool = True) -> numpy.ndarray:
        pass

    @abstractmethod
    def transform(self, x: numpy.ndarray) -> numpy.ndarray:
        pass


class Identity(Preprocessor):
    def fit(self, x: numpy.ndarray, transform: bool = True) -> numpy.ndarray:
        return x

    def transform(self, x: numpy.ndarray) -> numpy.ndarray:
        return x


class Standardizer(Preprocessor):
    def __init__(self):
        self.means: Union[numpy.ndarray, None] = None
        self.stds: Union[numpy.ndarray, None] = None
        self.__initialized = False

    def fit(self, x: numpy.ndarray, transform: bool = True) -> numpy.ndarray:
        self.means = x.mean(axis=0)
        self.stds = x.std(axis=1)
        self.__initialized = True
        return self.transform(x) if transform else x

    def transform(self, x: numpy.ndarray) -> numpy.ndarray:
        if not self.__initialized:
            raise RuntimeError('Standardizer must be initialized first! Use `fit`.')
        return (x - self.means) / self.means


class PCA(Preprocessor):
    def __init__(self):
        self.pca = SklearnPCA()

    def fit(self, x: numpy.ndarray, transform: bool = True) -> numpy.ndarray:
        self.pca.fit(x)
        return self.pca.transform(x) if transform else x

    def transform(self, x: numpy.ndarray) -> numpy.ndarray:
        return self.pca.transform(x)
