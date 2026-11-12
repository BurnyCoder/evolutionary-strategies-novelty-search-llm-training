"""Regression cases for independent read-only evidence verification (no model execution)."""

from dataclasses import asdict

import pytest

from novelty_es.data import grade
from novelty_es.verification import reference_grade, separate_output, verify_evaluation


@pytest.mark.parametrize('response', [
    '<answer>(2+3)*4</answer>', '<answer>2+3+4</answer>', '(2+3)\n*4',
    '(2+\n3)*4', '2**3+4', '2//3+4', '-2+3+4', '2+3+4+0',
    '<answer>2+3+4</answer><answer>2+3+4</answer>', 'prose <answer>(2+3)*4</answer> prose',
    '02+3+4', '2/(3-4)', '2/(3-3)', '',
])
def test_independent_parser_preserves_historical_python_syntax(response):
    assert reference_grade(response, [2, 3, 4], 20) == asdict(grade(response, [2, 3, 4], 20))


def test_output_must_be_outside_sources(tmp_path):
    for output in (tmp_path, tmp_path / 'raw', tmp_path / 'raw/summary'):
        with pytest.raises(ValueError, match='separate'):
            separate_output(output, [tmp_path / 'raw'])
    separate_output(tmp_path / 'derived', [tmp_path / 'raw'])


@pytest.mark.parametrize('rows', [[], [{'problem_id': 'wrong'}], [{'problem_id': 'x'}, {'problem_id': 'x'}]])
def test_incomplete_duplicate_and_unknown_evaluation_records_fail(rows):
    with pytest.raises(ValueError, match='evaluation|identity'):
        verify_evaluation({'sets': {'test': rows}}, {'test': [{'id': 'test-00000'}]}, {})
