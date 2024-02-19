import pandas as pd
from utils import build_grammar_from_df, build_frequent_set
import time
import os
from pyts.approximation import SymbolicAggregateApproximation
from sklearn.model_selection import train_test_split


eshopper_time = []
trainticket_time = []

eshopper_train, eshopper_test = train_test_split(
    os.listdir("../data/eshopper/"), test_size=0.2, random_state=42
)
train_data_eshopper = pd.DataFrame()
for file in eshopper_train:
    train_data_eshopper = pd.concat(
        [train_data_eshopper, pd.read_csv("data/eshopper/" + file)]
    )

train_ticket, test_ticket = train_test_split(
    os.listdir("../data/trainticket/"), test_size=0.2, random_state=42
)
train_data_ticket = pd.DataFrame()
for file in train_ticket:
    train_data_ticket = pd.concat(
        [train_data_ticket, pd.read_csv("../data/trainticket/" + file)]
    )


sax = SymbolicAggregateApproximation(n_bins=5)


for i in range(20):

    start_time = time.time()
    sax_train = sax.fit_transform(train_data_eshopper.drop(["anomaly"], axis=1).T).T
    build_frequent_set(sax_train[train_data_eshopper["anomaly"] == 1]).to_csv(
        "training_data_anomaly1.csv", index=False
    )
    build_frequent_set(sax_train[train_data_eshopper["anomaly"] == 2]).to_csv(
        "training_data_anomaly2.csv", index=False
    )
    parser1 = build_grammar_from_df("training_data_anomaly1.csv", "grammar_1")
    parser2 = build_grammar_from_df("training_data_anomaly2.csv", "grammar_2")
    end_time = time.time()

    eshopper_time.append(end_time - start_time)

    start_time = time.time()
    sax_train = sax.fit_transform(train_data_ticket.drop(["anomaly"], axis=1).T).T
    build_frequent_set(sax_train[train_data_ticket["anomaly"] == 1]).to_csv(
        "ticket_training_data_anomaly1.csv", index=False
    )
    build_frequent_set(sax_train[train_data_ticket["anomaly"] == 2]).to_csv(
        "ticket_training_data_anomaly2.csv", index=False
    )
    parser1 = build_grammar_from_df(
        "ticket_training_data_anomaly1.csv", "grammar_1_ticket"
    )
    parser2 = build_grammar_from_df(
        "ticket_training_data_anomaly2.csv", "grammar_2_ticket"
    )
    end_time = time.time()

    trainticket_time.append(end_time - start_time)

pd.DataFrame(
    {
        "E-Shopper": eshopper_time,
        "TrainTicket": trainticket_time,
    }
).to_csv("grammar_training_time.csv", index=False)
