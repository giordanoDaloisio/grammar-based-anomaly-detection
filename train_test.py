import pandas as pd
from pyts.approximation import SymbolicAggregateApproximation
import numpy as np
from sksequitur import Grammar, Parser, Production, Mark
import nltk
from nltk import CFG
from tqdm import tqdm
import re
import itertools
import time


def remove_adjacent_letters(input_string):
    # Define a regular expression pattern to match two or more adjacent letters
    pattern = re.compile(r"([a-z])\1")

    # Use the sub() function to replace matched patterns with an empty string
    result = re.sub(pattern, "", input_string)

    return result


def truncate_after_three_consecutive_spaces(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            # Find the index of three consecutive spaces
            index = line.find("   ")

            # If three consecutive spaces are found, truncate the line
            if index != -1:
                truncated_line = line[: index + 3] + "\n"
            else:
                truncated_line = line

            # Write the truncated line to the output file
            outfile.write(truncated_line)


def encapsulate_lowercase(file_path, output_file_path):
    with open(file_path, "r") as infile, open(output_file_path, "w") as outfile:
        for line in infile:
            modified_line = "".join(
                [f"'{char}'" if char.islower() else char for char in line]
            )
            outfile.write(modified_line)


def check_sentence(sentence):
    try:
        next(parser.parse(sentence))
        return 0
    except StopIteration:
        if (
            sentence[0] not in ["a", "b"]
            and sentence[2] not in ["a", "b"]
            and sentence[5] not in ["a", "b"]
        ):
            return 2
        return 1
    except ValueError:
        return 1


if __name__ == "__main__":
    frequent_items = pd.read_csv("training_data.csv")
    frequent_items = frequent_items[frequent_items["count"] > 2]

    parser = Parser()
    for item in frequent_items["string"]:
        parser.feed([Mark()])
        parser.feed(item)
    grammar = Grammar(parser.tree)

    with open("grammar_start.txt", "w") as f:
        print(grammar, file=f)

    input_file_path = "grammar_start.txt"
    middle_path = "grammar_middle.txt"
    output_file_path = "grammar.txt"

    truncate_after_three_consecutive_spaces(input_file_path, middle_path)

    encapsulate_lowercase(middle_path, output_file_path)

    with open("grammar.txt", "r") as f:
        grammar = f.read()

    grammar = CFG.fromstring(grammar)
    parser = nltk.ChartParser(grammar)

    test = pd.read_csv("test_data.csv")

    d = pd.DataFrame(test.drop(columns="anomaly"))
    d["result"] = d.apply(lambda x: check_sentence(x["string"]), axis=1)
    d["anomaly"] = test["anomaly"]

    d.to_csv("results.csv", index=False)
