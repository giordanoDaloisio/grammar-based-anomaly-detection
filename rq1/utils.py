import pandas as pd
import os
from sksequitur import Grammar, Parser, Mark
from nltk import CFG, ChartParser
import numpy as np


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


def build_grammar_from_df(path, grammar_name):
    frequent_items = pd.read_csv(path)
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
    output_file_path = f"{grammar_name}.txt"
    truncate_after_three_consecutive_spaces(input_file_path, middle_path)
    encapsulate_lowercase(middle_path, output_file_path)
    os.remove(input_file_path)
    os.remove(middle_path)
    with open(f"{grammar_name}.txt", "r") as f:
        grammar = f.read()
    grammar = CFG.fromstring(grammar)
    return ChartParser(grammar)


def check_sentence(sentence, parser1, parser2):
    try:
        next(parser1.parse(sentence))
        return 1
    except StopIteration:
        try:
            next(parser2.parse(sentence))
            return 2
        except StopIteration:
            return 0
    except ValueError:
        return 0
