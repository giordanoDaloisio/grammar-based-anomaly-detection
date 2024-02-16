import pandas as pd
from utils import build_grammar_from_df, check_sentence
import os

# E-Shopper

parser1 = build_grammar_from_df(
    os.path.join("eshopper", "training_data_anomaly1.csv"), "grammar_1"
)
parser2 = build_grammar_from_df(
    os.path.join("eshopper", "training_data_anomaly2.csv"), "grammar_2"
)

test = pd.read_csv(os.path.join("eshopper", "test_data.csv"))

test["result"] = test.apply(
    lambda x: check_sentence(x["string"], parser1, parser2), axis=1
)

test.to_csv("eshopper_result.csv", index=False)

# Train Ticket

parser1 = build_grammar_from_df(
    os.path.join("trainticket", "training_data_anomaly1.csv"), "grammar_1"
)
parser2 = build_grammar_from_df(
    os.path.join("trainticket", "training_data_anomaly2.csv"), "grammar_2"
)

test = pd.read_csv(os.path.join("trainticket", "test_data.csv"))

test["result"] = test.apply(
    lambda x: check_sentence(x["string"], parser1, parser2), axis=1
)

test.to_csv("ticket_result.csv", index=False)
