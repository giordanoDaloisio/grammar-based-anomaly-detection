import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import time

eshopper_time = []
trainticket_time = []

for i in range(20):

    train, test = train_test_split(
        os.listdir("data/eshopper/"), test_size=0.2, random_state=42
    )

    train_data = pd.DataFrame()
    for file in train:
        train_data = pd.concat([train_data, pd.read_csv("data/eshopper/" + file)])
    test_data = pd.DataFrame()
    for file in test:
        test_data = pd.concat([test_data, pd.read_csv("data/eshopper/" + file)])

    start_time = time.time()
    model = LogisticRegression()
    model.fit(train_data.drop(columns="anomaly"), train_data["anomaly"])
    end_time = time.time()

    predictions = model.predict(test_data.drop(columns="anomaly"))
    accuracy = accuracy_score(test_data["anomaly"], predictions)
    precision = precision_score(test_data["anomaly"], predictions, average="weighted")
    recall = recall_score(test_data["anomaly"], predictions, average="weighted")

    pd.DataFrame(
        {
            "Dataset": ["E-Shopper"],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
        }
    ).to_csv("metrics_eshopper.csv", index=False)

    eshopper_time.append(end_time - start_time)

    train, test = train_test_split(
        os.listdir("data/trainticket/"), test_size=0.2, random_state=42
    )
    train_data = pd.DataFrame()
    for file in train:
        train_data = pd.concat([train_data, pd.read_csv("data/trainticket/" + file)])

    test_data = pd.DataFrame()
    for file in test:
        test_data = pd.concat([test_data, pd.read_csv("data/trainticket/" + file)])

    start_time = time.time()
    model = LogisticRegression()
    model.fit(train_data.drop(columns="anomaly"), train_data["anomaly"])
    end_time = time.time()

    trainticket_time.append(end_time - start_time)

    predictions = model.predict(test_data.drop(columns="anomaly"))
    accuracy = accuracy_score(test_data["anomaly"], predictions)
    precision = precision_score(test_data["anomaly"], predictions, average="weighted")
    recall = recall_score(test_data["anomaly"], predictions, average="weighted")

    pd.DataFrame(
        {
            "Dataset": ["Train Ticket"],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
        }
    ).to_csv("metrics_trainticket.csv", index=False)

ttimes = pd.DataFrame(
    {"E-Shopper": eshopper_time, "Train Ticket": trainticket_time}
).to_csv("ml_training_time.csv", index=False)
