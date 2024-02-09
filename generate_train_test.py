import pandas as pd
import numpy as np
import os
from pyts.approximation import SymbolicAggregateApproximation
from sklearn.model_selection import train_test_split


def build_frequent_set(data):
    joined = np.apply_along_axis(lambda x: "".join(x), 1, data)
    string, count = np.unique(joined, return_counts=True)
    return pd.DataFrame({"string": string, "count": count})


if __name__ == "__main__":
    train, test = train_test_split(
        os.listdir("data/trainticket/"), test_size=0.2, random_state=42
    )
    train_data = pd.DataFrame()
    for file in train:
        train_data = pd.concat([train_data, pd.read_csv("data/trainticket/" + file)])
    test_data = pd.DataFrame()
    for file in test:
        test_data = pd.concat([test_data, pd.read_csv("data/trainticket/" + file)])

    sax = SymbolicAggregateApproximation(n_bins=5)
    sax_train = sax.fit_transform(train_data.drop(["anomaly"], axis=1).T).T

    build_frequent_set(sax_train[train_data["anomaly"] == 0]).to_csv(
        "ticket_training_data_no_anomaly.csv", index=False
    )
    build_frequent_set(sax_train[train_data["anomaly"] == 1]).to_csv(
        "ticket_training_data_anomaly1.csv", index=False
    )
    build_frequent_set(sax_train[train_data["anomaly"] == 2]).to_csv(
        "ticket_training_data_anomaly2.csv", index=False
    )

    sax_test = sax.transform(test_data.drop(["anomaly"], axis=1).T).T
    sax_data_test = np.apply_along_axis(lambda x: "".join(x), 1, sax_test)
    df_test = pd.DataFrame({"string": sax_data_test, "anomaly": test_data["anomaly"]})
    df_test.to_csv("ticket_test_data.csv", index=False)
