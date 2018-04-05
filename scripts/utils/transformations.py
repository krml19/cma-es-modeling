from sklearn.preprocessing import StandardScaler


class StandardTransformer:
    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.__standard_scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)

    def fit(self, X):
        self.__standard_scaler.fit(X)

    def transform(self, X):
        return self.__standard_scaler.transform(X)

    def inverse_transform(self, X):
        return self.__standard_scaler.inverse_transform(X)