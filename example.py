from codecarbon import EmissionsTracker
from e2efs_pytorch.dataset_reader import dexter, gina_v2 as gina, gisette, madelon, colon, leukemia, lung181, lymphoma
import numpy as np
from e2efs_pytorch import e2efs
from sklearn.model_selection import RepeatedStratifiedKFold
import json
import os
import torch
path = os.path.dirname(os.path.realpath(__file__))

n_features_to_select = [10,
                        20,
                        40,
                        80]
datasets = [#(dexter, "dexter"),
            (gina, "gina"),]
            #(gisette, "gisette"),
            #(madelon, "madelon"),
            #(colon, "colon"),
            #(leukemia, "leukemia"),
            #(lung181, "lung181"),
            #(lymphoma, "lymphoma")]
precisions = ["16-mixed",
              "32",
              "64"]
results = {
    "dataset": [],
}

if __name__ == '__main__':

    for ds in datasets:
        for prec in precisions:
            #recomendado por pytorch
            if prec == "16-mixed":
                torch.set_float32_matmul_precision("medium")
            else:
                #para precisiones mayores a float16, se establece a float32 el tipo de dato
                torch.set_float32_matmul_precision("highest")
            for n_feat in n_features_to_select:
                print ("\n", ds[1], prec, n_feat, "\n")
                tracker = EmissionsTracker(
                    log_level="warning",
                    output_file= path + "/results/emissions_" + ds[1] + "_" + prec + "_" + str(n_feat) + ".csv")
                tracker.start()
                dataset = ds[0].load_dataset()
                raw_data = np.asarray(dataset['raw']['data'])
                raw_label = np.asarray(dataset['raw']['label']).reshape(-1)

                rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5,
                                            random_state=36851234)

                bas = []
                accs = []

                for i, (train_index, test_index) in enumerate(rskf.split(raw_data, raw_label)):
                    train_data, train_label = raw_data[train_index], raw_label[train_index]
                    test_data, test_label = raw_data[test_index], raw_label[test_index]

                    normalize = ds[0].Normalize()
                    train_data = normalize.fit_transform(train_data)
                    test_data = normalize.transform(test_data)

                    batch_size = max(2, len(train_data)//50)

                    # Allowed precision values: ('transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', 64, 32, 16, '64', '32', '16', 'bf16')
                    # Con 16-true non funciona
                    model = e2efs.E2EFSSoft(n_features_to_select=n_feat, precision=prec, network='three_layer_nn')
                    # model = e2efs.E2EFSRanking()
                    model.fit(train_data, train_label, batch_size=batch_size, max_epochs=5000, validation_data=(test_data, test_label))
                    model.fine_tune(train_data, train_label, validation_data=(test_data, test_label), batch_size=batch_size, max_epochs=100)

                    metrics = model.evaluate(test_data, test_label)
                    print(metrics)
                    print(i)
                    bas.append(metrics['test_ba'])
                    accs.append(metrics['test_accuracy'])

                results["dataset"].append({
                    "name": ds[1],
                    "precision": prec,
                    "n_features": n_feat,
                    "ba": bas,
                    "acc": accs
                    })

                print("Dataset:", ds[1])
                print("Precision:", prec)
                print("N features:", n_feat)
                print('Global BA:', np.mean(bas))
                print('Global ACC:', np.mean(accs))
                tracker.stop()
                json.dump(results, open(path + '/results/results.json', 'w'))