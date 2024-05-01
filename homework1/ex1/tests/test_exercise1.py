import pytest
import math
import numpy as np
import exercise1




def test_calculate_Q3_result_type():
    result = exercise1.calculate_Q3("HHCHEHHEH", "HECHECHEC")
    assert isinstance(exercise1.calculate_Q3("HHCHEHHEH", "HECHECHEC"), tuple) 
    
    
def test_calculate_Q3_result_values():
    result = exercise1.calculate_Q3("HHHEEECCC", "HHHEECHEC")
    # avoid floating number artifacts
    assert int(result[0]*100) == 75 and int(result[1]*100) == 66 and int(result[2]*100) == 50
    
    
def test_calculate_Q3_missing_states():
    result = exercise1.calculate_Q3("HHHEEECCC", "HHHEEHHEH")
    # avoid floating number artifacts
    assert math.isnan(result[2])
    
def test_load_jsonl_path():
    input_file = "./tests/data.jsonl"
    entries = exercise1.load_jsonl(input_file)
    assert len(entries) != 0
    
def test_load_jsonl_entry_numbers():
    input_file = "./tests/data.jsonl"
    entries = exercise1.load_jsonl(input_file)
    assert len(entries) == 10

def test_load_jsonl_types():
    input_file = "./tests/data.jsonl"
    entries = exercise1.load_jsonl(input_file)
    assert isinstance(entries, list) and isinstance(entries[0], dict)
    
def test_single_one_hot_encode_input_length():
    with pytest.raises(ValueError):
        exercise1.single_one_hot_encode("AA")
    
def test_single_one_hot_encode_input_accepted_aa():
    with pytest.raises(ValueError):
        exercise1.single_one_hot_encode("Z")

def test_single_one_hot_encode_input_correct_embedding_values():
    assert np.array_equal(exercise1.single_one_hot_encode("C"), np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      dtype=np.int8))
    
def test_single_one_hot_encode_input_correct_embedding_type():
    assert exercise1.single_one_hot_encode("C").dtype == np.int8
    
def test_one_hot_encode_sequence_shape():
    sequence = "AAACYYY"
    window_size = 3

    assert exercise1.one_hot_encode_sequence(sequence, window_size).shape == (1,140)

def test_one_hot_encode_sequence_values():
    sequence = "AAACYYY"
    window_size = 2
    
    oracle = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.int8)

    assert np.array_equal(exercise1.one_hot_encode_sequence(sequence, window_size), oracle)
    
    
def test_one_hot_encode_labeled_sequence_encoding_rows():
    demo_entry = {'sequence' : "AAACYYY", 'label':'HHHCEEE', 'resolved':'0010100'}
    window_size = 2
    assert exercise1.one_hot_encode_labeled_sequence(demo_entry, window_size)[0].shape[0] == 2 
    

def test_one_hot_encode_labeled_sequence_columns():
    demo_entry = {'sequence' : "AAACYYY", 'label':'HHHCEEE', 'resolved':'0010100'}
    window_size = 2
    assert exercise1.one_hot_encode_labeled_sequence(demo_entry, window_size)[0].shape[1] == 100 

def test_one_hot_encode_labeled_sequence_encoding_values():
    demo_entry = {'sequence' : "AAACYYY", 'label':'HHHCEEE', 'resolved':'0010100'}
    window_size = 2

    oracle = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.int8)
    assert np.array_equal(exercise1.one_hot_encode_labeled_sequence(demo_entry, window_size)[0],oracle)

def test_one_hot_encode_labeled_sequence_labels():
    demo_entry = {'sequence' : "AAACYYY", 'label':'HHHCEEE', 'resolved':'0010100'}
    window_size = 2
    oracle = np.array([0,1], dtype=np.int8)
    assert np.array_equal(exercise1.one_hot_encode_labeled_sequence(demo_entry, window_size)[1],oracle)
