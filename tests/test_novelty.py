"""Reference examples distinguish correctness gating, sibling exclusion and history."""

import pytest
import torch

from novelty_es.novelty import CorrectMemory


def test_empty_memory_and_wrong_answer_gate():
    memory = CorrectMemory(6, 0)
    emb = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    correct = torch.tensor([[True], [False]])
    assert torch.equal(memory.score(emb, correct), torch.tensor([[1.0], [0.0]]))
    assert memory.history == {}
    memory.append(emb, correct)
    assert torch.equal(memory.history[0], emb[0])
    assert torch.equal(memory.score(emb, correct), torch.zeros(2, 1))


def test_siblings_and_cosine_are_not_euclidean_distance():
    memory = CorrectMemory(6, 0)
    emb = torch.tensor([[[2.0, 0.0]], [[0.0, 3.0]], [[2.0, 2.0]]])
    score = memory.score(emb, torch.ones(3, 1, dtype=torch.bool))
    assert torch.allclose(score, torch.full((3, 1), 1 - 2**-0.5), atol=1e-6)
    assert memory.history == {}  # Scoring must not mutate the frozen archive.


def test_duplicate_sibling_zero_and_negative_cosine():
    memory = CorrectMemory(6, 0)
    correct = torch.ones(2, 1, dtype=torch.bool)
    assert torch.equal(memory.score(torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]]), correct), torch.zeros(2, 1))
    assert torch.equal(memory.score(torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]]), correct), torch.full((2, 1), 2.0))


def test_memory_bound_shuffle_and_resume():
    emb = torch.arange(1.0, 61.0).reshape(10, 2, 3)
    correct = torch.ones(10, 2, dtype=torch.bool)
    a, b = CorrectMemory(6, 9), CorrectMemory(6, 9)
    a.append(emb, correct)
    b.load_state_dict(a.state_dict())
    a.append(emb.flip(0), correct)
    b.append(emb.flip(0), correct)
    for p in a.history:
        assert a.history[p].shape == (6, 3)
        assert torch.equal(a.history[p], b.history[p])


def test_gate_must_be_boolean_not_reward():
    with pytest.raises(TypeError):
        CorrectMemory(6, 0).score(torch.ones(2, 1, 2), torch.tensor([[1.0], [0.1]]))
