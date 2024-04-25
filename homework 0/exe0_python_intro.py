from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

AAs: str = 'ALGVSREDTIPKFQNYMHWCUZO'


def get_aa_composition(protein_seq: str) -> Dict[str, int]:
    aa_composition = {}
    for aa in protein_seq:
        if aa in aa_composition:
            aa_composition[aa] += 1
        else:
            aa_composition[aa] = 1
    return aa_composition


def k_mers(alphabet: str, k: int) -> List[str]:
    if k == 1:
        return list(alphabet)
    else:
        kmers = []
        for kmer in k_mers(alphabet, k - 1):
            for aa in alphabet:
                kmers.append(kmer + aa)
        return kmers


def get_kmer_composition(protein_seq: str, k: int) -> Dict[str, int]:
    kmer_dict = defaultdict(int)  # Initialize a dictionary to store k-mer occurrences
    seq_len = len(protein_seq)
    
    # Iterate through the protein sequence
    for i in range(seq_len - k + 1):
        kmer = protein_seq[i:i+k]  # Extract the current k-mer
        kmer_dict[kmer] += 1  # Increment the occurrence count for the current k-mer
    
    # Add missing k-mers with 0 occurrence
    for i in range(0, 20 ** k):
        kmer = ''.join([chr((i // (20 ** j)) % 20 + 65) for j in range(k)])
        if kmer not in kmer_dict:
            kmer_dict[kmer] = 0
    
    return kmer_dict

def get_alignment(protein_seq_1: str, protein_seq_2: str,
                  gap_penalty: int, substitution_matrix: Dict[str, Dict[str, int]]) -> Tuple[str, str]:
    return '', ''
