import pandas as pd
from utils import build_grammar_from_df, check_sentence

parser1 = build_grammar_from_df("training_data_anomaly1.csv", "grammar_1")
parser2 = build_grammar_from_df("training_data_anomaly2.csv", "grammar_2")

test = pd.read_csv("test_data.csv")

test["result"] = test.apply(
    lambda x: check_sentence(x["string"], parser1, parser2), axis=1
)

test.to_csv("result_new.csv", index=False)
