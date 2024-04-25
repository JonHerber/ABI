import json
import pytest
import numpy as np

from pathlib import Path
from .matrices import MATRICES
from tests import *


@pytest.fixture(scope="module")
def json_data():
    test_json = 'exe0_test.json'
    relative_path = Path(__file__).parent

    with Path(relative_path, test_json).open('r') as json_file:
        json_data = json.load(json_file)

    return json_data


@pytest.fixture(scope="module")
def exe0_aa_comp_test(json_data):
    return json_data['exe0_aa_comp_test']


@pytest.fixture(scope="module")
def exe0_aa_comp_test_correct(exe0_aa_comp_test):
    return exe0_aa_comp_test['correct']


@pytest.fixture(scope="module")
def exe0_aa_comp_test_wrong_1(exe0_aa_comp_test):
    return exe0_aa_comp_test['wrong_seq_1']


@pytest.fixture(scope="module")
def exe0_aa_comp_test_wrong_2(exe0_aa_comp_test):
    return exe0_aa_comp_test['wrong_seq_2']


@pytest.fixture(scope="module")
def test_exe0_aa_comp_test_wrong_1(exe0_aa_comp_test_wrong_1):
    try:
        from exe0_python_intro import get_aa_composition
        student = get_aa_composition(exe0_aa_comp_test_wrong_1['sequence'])
    except ValueError:
        return True, ''
    except Exception:
        raise AssertionError('Error in exe0_python_intro.get_aa_composition().') from None

    return False, ''


def test_exe0_aa_comp_wrong_1(test_exe0_aa_comp_test_wrong_1):
    passed, assertion_msg, *_ = test_exe0_aa_comp_test_wrong_1
    assert passed, f'Failed test_exe0_aa_comp_wrong_1(). {assertion_msg}'


@pytest.fixture(scope="module")
def test_exe0_aa_comp_test_wrong_2(exe0_aa_comp_test_wrong_2):
    try:
        from exe0_python_intro import get_aa_composition
        student = get_aa_composition(exe0_aa_comp_test_wrong_2['sequence'])
    except ValueError:
        return True, ''
    except Exception:
        raise AssertionError('Error in exe0_python_intro.get_aa_composition().') from None

    return False, ''


def test_exe0_aa_comp_wrong_2(test_exe0_aa_comp_test_wrong_2):
    passed, assertion_msg, *_ = test_exe0_aa_comp_test_wrong_2
    assert passed, f'Failed test_exe0_aa_comp_wrong_2(). {assertion_msg}'


@pytest.fixture(scope="module")
def test_exe0_aa_comp_test_correct(exe0_aa_comp_test_correct):
    try:
        from exe0_python_intro import get_aa_composition
        student = get_aa_composition(exe0_aa_comp_test_correct['sequence'])
    except Exception:
        raise AssertionError('Error in exe0_python_intro.get_aa_composition().') from None

    try:
        passed = isinstance(student, dict)
        assert passed, 'Return expected to be dict.'
        passed = (student == exe0_aa_comp_test_correct['out'])
        assert passed, 'Computation is wrong.'
    except AssertionError as msg:
        return False, msg

    return True, ''


def test_exe0_aa_comp_correct(test_exe0_aa_comp_test_correct):
    passed, assertion_msg, *_ = test_exe0_aa_comp_test_correct
    assert passed, f'Failed test_exe0_aa_comp_correct(). {assertion_msg}'


@pytest.fixture(scope="module")
def exe0_kmer_test(json_data):
    return json_data['exe0_kmer_test']


@pytest.fixture(scope="module")
def test_exe0_kmer_test(exe0_kmer_test):
    try:
        from exe0_python_intro import k_mers
        student = k_mers(exe0_kmer_test['alphabet'], exe0_kmer_test['k'])
    except Exception:
        raise AssertionError('Error in exe0_python_intro.k_mers().') from None

    try:
        # TODO check for lists and the lenth matching
        passed = (set(student) == set(exe0_kmer_test['out']))
        assert passed, 'Computation is wrong.'
    except AssertionError as msg:
        return False, msg

    return True, ''


def test_exe0_kmer(test_exe0_kmer_test):
    passed, assertion_msg, *_ = test_exe0_kmer_test
    assert passed, f'Failed test_exe0_kmer(). {assertion_msg}'


@pytest.fixture(scope="module")
def exe0_kmer_comp_test(json_data):
    return json_data['exe0_kmer_comp_test']


@pytest.fixture(scope="module")
def test_exe0_kmer_comp_test(exe0_kmer_comp_test):
    try:
        from exe0_python_intro import get_kmer_composition
        student = get_kmer_composition(exe0_kmer_comp_test['sequence'], exe0_kmer_comp_test['k'])
    except Exception:
        raise AssertionError('Error in exe0_python_intro.get_kmer_composition().') from None

    try:
        passed = (student == exe0_kmer_comp_test['out'])
        assert passed, 'Computation is wrong.'
    except AssertionError as msg:
        return False, msg

    return True, ''


def test_exe0_kmer_comp(test_exe0_kmer_comp_test):
    passed, assertion_msg, *_ = test_exe0_kmer_comp_test
    assert passed, f'Failed test_exe0_kmer_comp(). {assertion_msg}'


@pytest.fixture(scope="module")
def exe0_align_test(json_data):
    return json_data['exe0_align_test']


@pytest.fixture(scope="module")
def exe0_align_test_null(exe0_align_test):
    return exe0_align_test['null']


@pytest.fixture(scope="module")
def exe0_align_test_short(exe0_align_test):
    return exe0_align_test['short']


def create_LA(string_1, string_2, gap_penalty, matrix):
    try:
        from exe0_python_intro import get_alignment
        la = get_alignment(string_1, string_2, gap_penalty, matrix)
        return isinstance(la, tuple), 'Alignment should be of type tuple!', la
    except Exception:
        return False, 'Error while exe0_python_intro.get_alignment().', None


@pytest.fixture(scope="module")
def null_la(exe0_align_test_null):
    passed, assertion_msg, null_la = create_LA(*exe0_align_test_null['strings'],
                                               exe0_align_test_null['gap_penalty'],
                                               MATRICES[exe0_align_test_null['matrix']])
    assert passed, assertion_msg
    return null_la


@pytest.fixture(scope="module")
def short_la(exe0_align_test_short):
    passed, assertion_msg, short_la = create_LA(*exe0_align_test_short['strings'],
                                                exe0_align_test_short['gap_penalty'],
                                                MATRICES[exe0_align_test_short['matrix']])
    assert passed, assertion_msg
    return short_la


def check_alignment(alignment, case, case_text):
    try:
        passed = isinstance(alignment, tuple)
        assert passed, 'Return type is not a tuple.'

        passed = len(alignment) == 2
        assert passed, f'Tuple contains {len(alignment)} elements. Expected: 2'

        passed = all([isinstance(s, str) for s in alignment])
        assert passed, 'Tuple does not contain only strings.'

        passed = (tuple(case['alignment']) == alignment)
        return passed, f'Incorrect alignment ({case_text} strings).'
    except AssertionError as msg:
        return False, msg


@pytest.fixture(scope="module")
def get_alignment_on_small_strings(short_la, exe0_align_test_short):
    try:
        return check_alignment(short_la, exe0_align_test_short, 'short')
    except Exception:
        raise AssertionError('Error in exe0_python_intro.get_alignment().') from None


def test_get_alignment_on_small_strings(get_alignment_on_small_strings):
    passed, assertion_msg, *_ = get_alignment_on_small_strings
    assert passed, f'Failed test get_alignment_on_small_strings(). {assertion_msg}'


@pytest.fixture(scope="module")
def get_alignment_on_null_strings(null_la, exe0_align_test_null):
    try:
        return check_alignment(null_la, exe0_align_test_null, 'null')
    except Exception:
        raise AssertionError('Error in LocalAlignment.get_alignment().') from None


def test_get_alignment_on_null_strings(get_alignment_on_null_strings):
    passed, assertion_msg, *_ = get_alignment_on_null_strings
    assert passed, f'Failed test get_alignment_on_null_strings(). {assertion_msg}'

