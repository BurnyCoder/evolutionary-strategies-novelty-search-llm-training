"""Continuation checks use synthetic CPU fixtures, never reported model results."""

from novelty_es.data import generate_dataset, grade, load_data, prepare_data


def test_fresh_data_excludes_every_parent_multiset(tmp_path):
    """A fresh endpoint excludes all earlier splits, including permutations and targets."""
    parent = generate_dataset(11, {"train": 40, "test": 20, "hard": 10})
    excluded = {tuple(sorted(row["numbers"])) for rows in parent.values() for row in rows}
    manifest = prepare_data(tmp_path, 11, {"fresh": 24}, excluded)
    fresh = load_data(tmp_path)["fresh"]
    assert manifest["splits"]["fresh"]["count"] == 24
    assert not excluded & {tuple(sorted(row["numbers"])) for row in fresh}
    assert sum(len(row["numbers"]) == 3 for row in fresh) == 12
    assert all(grade(row["witness"], row["numbers"], row["target"]).correct for row in fresh)
    assert generate_dataset(11, {"fresh": 24}, excluded)["fresh"] == fresh
