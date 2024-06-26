from e2efs_pytorch.dataset_reader import antonio
import numpy as np
from e2efs_pytorch import e2efs
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFE
from sklearn.svm import SVR


n_features_to_select = 10
maes = []
mses = []

if __name__ == '__main__':

    dataset = antonio.load_dataset()
    raw_data = np.asarray(dataset['raw']['data'])
    raw_target = np.asarray(dataset['raw']['target']).reshape(-1)

    rskf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=36851234)

    for i, (train_index, test_index) in enumerate(rskf.split(raw_data, raw_target)):
        train_data, train_target = raw_data[train_index], raw_target[train_index]
        test_data, test_target = raw_data[test_index], raw_target[test_index]

        normalize = antonio.Normalize()
        train_data = normalize.fit_transform(train_data)
        test_data = normalize.transform(test_data)

        # Allowed precision values: ('transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', 64, 32, 16, '64', '32', '16', 'bf16')
        # Con 16-true non funciona
        estimator = SVR(kernel="linear")
        model = RFE(estimator, n_features_to_select=15, step=1)
        # model = e2efs.E2EFSRanking()
        model.fit(train_data, train_target)
        test_predict = model.predict(test_data)

        metrics = {
            'test_mse': np.mean(np.square(test_predict - test_target)),
            'test_mae': np.mean(np.abs(test_predict - test_target)),
        }
        print(metrics)
        mses.append(metrics['test_mse'])
        maes.append(metrics['test_mae'])

    print('Global MSE:', np.mean(mses))
    print('Global MAE:', np.mean(maes))
