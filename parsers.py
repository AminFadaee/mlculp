from abc import ABC, abstractmethod

import numpy


class Parser(ABC):
    number_of_features: int
    number_of_classes: int

    def __init__(self, data: str):
        self.data = data
        self._x = []
        self._y = []
        self.parse()

    @abstractmethod
    def parse(self):
        pass

    @property
    @abstractmethod
    def x(self):
        pass

    @property
    @abstractmethod
    def y(self):
        pass


class MekaParser(Parser):
    def parse(self):
        lines = self.data.split('\n')
        index_of_data = lines.index('@data') + 2
        actual_data = lines[index_of_data:]

        self.number_of_classes = self._parse_number_of_classes(lines[0])
        self.number_of_features = index_of_data - self.number_of_classes - 5

        for line in actual_data:
            if not line or line == '\n':
                continue
            self._y.append([0] * self.number_of_classes)
            self._x.append([0] * self.number_of_features)
            self._parse_line(line, self._x[-1], self._y[-1])

    def _parse_line(self, line, x_list, y_list):
        bow = False
        if line.startswith('{'):
            bow = True
            line = line[1:-1]
        features = line.split(',')
        for index, feature in enumerate(features):
            if bow:
                attribute, value = map(int, feature.split(' '))
            else:
                attribute = index
                value = float(feature)
            if attribute < self.number_of_classes:
                y_list[attribute] = value
            else:
                x_list[attribute - self.number_of_classes] = value

    def _parse_number_of_classes(self, first_line):
        index = first_line.find('-C ') + 3
        number_of_classes = int(first_line[index:].split(' ')[0].split("'")[0])
        return number_of_classes

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


def read_scene_data():
    file = open('data/Scene.arff')
    content = file.read()
    file.close()
    parser = MekaParser(content)

    X = numpy.asarray(parser.x, dtype=numpy.float64)
    Y = numpy.asarray(parser.y, dtype=int)
    return X, Y
