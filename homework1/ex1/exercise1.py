import numpy as np
import json
from typing import Dict, List, Tuple
import os, sys

AMINO_ACIDS = "ACEDGFIHKMLNQPSRTWVY"

#input_file = "./data/input.jsonl"
def load_jsonl(filename: str)-> List[Dict]:
    input = open(os.path.join(filename), 'r')
    elements = input.readlines()
    protein_list = []
    for line in elements:
        line = line.split(",")
        protein = {}
        for content in line:
            content = content.split(":")
            protein[content[0].split('"')[1]] = content[1].split('"')[1]
        #filter X
        print(protein)
        if 'X' in protein["sequence"]:
            x_pos = list(protein["sequence"]).index('X')
            protein["sequence"] = protein["sequence"][:x_pos] + protein["sequence"][x_pos+1:]
            protein["label"] = protein["label"][:x_pos] + protein["label"][x_pos+1:]
            protein["resolved"] = protein["resolved"][:x_pos] + protein["resolved"][x_pos+1:]


        protein_list.append(protein)

    return protein_list
def single_one_hot_encode(amino_acid: "str") -> np.array:
    if len(amino_acid) != 1 or amino_acid not in AMINO_ACIDS:
        raise ValueError("Can`t encode")
    this_amino = np.zeros(20, dtype=np.int8)
    this_amino[list(AMINO_ACIDS).index(amino_acid)] = 1

    return this_amino

def one_hot_encode_sequence(sequence:str, window_size=5) -> np.array:
    '''This function takes a sequence of amino acids and converts it into a 2D Numpy array
    representing the one-hot encoding. It has len(sequence) - 2*window_size rows and 20*(window_size*2 + 1) columns.
    '''
    pass

def one_hot_encode_labeled_sequence(entry: Dict, window_size=5) -> Tuple[np.array, np.array]:
    '''This function takes an entry dict containing the sequence, the label and the resolved information
    and returns as first component of the tuple the one-hot encoding for every residue including its sliding window that has:
    - a label
    - enough neighboring residues to fill the sliding window.
    The second component of the tuple is a Numpy array that contains the respective labels encoded as 0,1,2 for H,E,C.
    Remember: Both arrays have to have the same length; In this case internal unresolved residues should be considered
    and excluded from the encoding.
    '''
    pass

def predict_secondary_structure(input: np.array, labels:np.array, size_hidden=10) -> Tuple[float, float, float]:
    '''This function creates a sklearn.neural_network.MLPClassifier objects with all defaults except hidden_layer_sizes is
    set to (size_hidden,) and with random_state set to 42'''
    pass


def calculate_Q3(prediction: str, truth:str) -> Tuple[float,float, float]:
    '''Compares two strings of equal length:
    prediction: string of predicted states H/E/C
    truth: string of true states H/E/C
    both strings are of the same length
    returns the fraction of correct predictions for every state (H/E/C) as a 3-tuple
    '''
    pass


if __name__ == "__main__":
    input_file = "./data/input.jsonl"
    entries = load_jsonl(input_file)
    print(len(entries))

    # extend as you need
    encoded_amino_acid = single_one_hot_encode('Y')
    print(encoded_amino_acid)
    pass
