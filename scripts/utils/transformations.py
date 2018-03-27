import numpy as np


class Transformation:

    def operation(self, array: np.ndarray) -> np.ndarray:
        pass

    def inverse_operation(self, array: np.ndarray) -> np.ndarray:
        pass


class StandardizationTransformation(Transformation):
    __mean = 0
    __std = 0

    def operation(self, array: np.ndarray):
        assert isinstance(array, np.ndarray)
        self.__mean = array.mean()
        self.__std = array.std()

        return np.vectorize(lambda x: (x - self.__mean) / self.__std, array)

    def inverse_operation(self, array: np.ndarray):
        assert isinstance(array, np.ndarray)
        return np.vectorize(lambda x: x * self.__std + self.__mean, array)