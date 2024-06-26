import numpy as np
from scipy.special import erf
import pandas as pd 
import os


data_filename = 'CSV_T3.csv'

def load_dataset(directory=None):
    directory = (os.path.dirname(os.path.realpath(__file__)) + '/../datasets' if directory is None else directory) + '/antonio/'
    dataset = load_data(directory)
    return dataset


def load_data(directory):
    info = {
        'raw': {}
    }

    df_T3 = pd.read_csv(directory + data_filename)
    df_T3['RF'] = df_T3['RF'].fillna('unknown')
    df_T3['smoking'] = df_T3['smoking'].fillna('UNK')
    dummies_df_T3 = pd.get_dummies(df_T3, drop_first=True)
    dummies_df_T3_sin_NaN = dummies_df_T3.dropna(subset=["DAST3-T0"])
    dummies_df_T3_sin_NaN = dummies_df_T3_sin_NaN.fillna(dummies_df_T3_sin_NaN.mean())
    # dummies_df_T3_sin_NaN = dummies_df_T3.dropna()
    X_T3 = dummies_df_T3_sin_NaN.drop('DAST3-T0', axis=1)
    y_T3 = dummies_df_T3_sin_NaN['DAST3-T0']

    info['raw']['data'] = X_T3.values
    info['raw']['target'] = y_T3.values.reshape((-1, 1))

    return info


class Normalize:

    def __init__(self):
        self.stats = None
        self.val_Stats = None

    def fit(self, X):
        X_mean = np.mean(X, axis=0)
        X_std = np.sqrt(np.square(X - X_mean).sum(axis=0) / max(1, len(X) - 1))
        self.stats = (X_mean, X_std)

    def transform(self, X):
        # transformed_X = erf((X - self.stats[0]) / (np.maximum(1e-6, self.stats[1]) * np.sqrt(2.)))
        transformed_X = (X - self.stats[0]) / (np.maximum(1e-6, self.stats[1]))
        return transformed_X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def val_fit(self, X):
        X_mean = np.mean(X, axis=0)
        X_std = np.sqrt(np.square(X - X_mean).sum(axis=0) / max(1, len(X) - 1))
        self.val_stats = (X_mean, X_std)

    def val_transform(self, X):
        # transformed_X = erf((X - self.stats[0]) / (np.maximum(1e-6, self.stats[1]) * np.sqrt(2.)))
        transformed_X = (X - self.val_stats[0]) / (np.maximum(1e-6, self.val_stats[1]))
        return transformed_X

    def val_fit_transform(self, X):
        self.val_fit(X)
        return self.val_transform(X)
