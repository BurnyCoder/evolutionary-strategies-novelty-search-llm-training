"""Check regenerated numbers and historical chronology using existing records only.

This is complementary to novelty-es verify: no old audit verdict is an input.
NumPy percentile resampling: https://numpy.org/doc/stable/reference/random/generator.html .
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from novelty_es.verification import read, separate_output, sha256


def interval(values, replicates):
    """Resample raw aligned values independently of the reporting implementation."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261107)
    means = np.concatenate([values[rng.integers(len(values), size=(min(256, replicates - i), len(values)))].mean(1)
                            for i in range(0, replicates, 256)])
    low, high = np.quantile(means, [0.025, 0.975])
    return dict(difference=float(values.mean()), low=float(low), high=float(high), n=len(values))


def equal(left, right):
    """Tolerate only roundoff in means; keys, counters and strings must match."""
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, float):
        return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
    return left == right


def comparison(a, b, split, replicates):
    first, second = ({row['problem_id']: int(row['correct']) for row in r['sets'][split]} for r in (a, b))
    assert first.keys() == second.keys()
    return interval([first[key] - second[key] for key in sorted(first)], replicates)


def check_study(directory, release, verified):
    """Check every empirical summary field used by tables, plots and examples."""
    experiment = read(directory / 'experiment.json')
    config = experiment['config']
    slug = 'continuation-seed0-gen100' if 'continuation' in experiment else 'pilot'
    summary = read(release / 'releases' / slug / 'summary.json')
    raw = {name: read(directory / name / 'evaluation.json') for name in summary['methods']}
    reps, split = config['bootstrap_replicates'], summary['primary_split']
    for name, method in summary['methods'].items():
        for label, rows in raw[name]['sets'].items():
            actual, reference = method[label], verified['evaluations'][name][label]
            assert actual['generated_tokens'] == reference['generated_tokens']
            if label == 'sampled':
                for key in ('pass1', 'pass8', 'mean_unique_correct_structures'):
                    assert equal(actual[key], reference[key]), (name, label, key)
                assert actual['complete_problems'] == config['sampled_problem_count']
                assert actual['partial_problems'] == 0
            else:
                count = len(rows)
                assert actual['n'] == reference['responses'] == actual['expected']
                for key, value in dict(accuracy=reference['correct'] / count, validity=reference['valid'] / count,
                                       reward=(reference['correct'] + .1 * (reference['valid'] - reference['correct'])) / count,
                                       truncated=reference['truncated'], correct_expression_structures=reference['correct']).items():
                    assert equal(actual[key], value), (name, label, key)
                ci = interval([int(row['correct']) for row in rows], reps)
                assert equal(actual['interval'], [ci['low'], ci['high']])
        assert summary['examples'][name] == raw[name]['sets'][split][:16]
        metrics_path = directory / name / 'metrics.json'
        metrics = read(metrics_path) if metrics_path.exists() else []
        for key in ('responses', 'generated_tokens', 'truncated'):
            assert method['training_observations'][key] == sum(row[key] for row in metrics)
            first = experiment.get('continuation', {}).get('parent_generation', 0)
            assert method['added_training_observations'][key] == sum(row[key] for row in metrics[first:])
        if name != 'original':
            expected = verified['trace']['coefficients'][name]
            diagnostic = method['training_observations']['weight_diagnostic']
            for key in ('candidates_weight_changed', 'candidates_positive_novelty', 'largest_absolute_weight_change'):
                assert equal(diagnostic[key], expected[key]), key
            assert equal(summary['baseline_comparisons'][name], comparison(raw[name], raw['original'], split, reps))
    assert equal(summary['primary'], verified['primary'])
    logged = {f"{row['condition']}/{row['phase']}": {k: row[k] for k in ('responses', 'generated_tokens')}
              for row in summary['logged_generation_work']}
    assert logged == verified['trace']['response_events']
    if 'continuation' in experiment:
        gains = {}
        for name in ('main', 'reward_only'):
            parent = read(directory / f'parent_{name}' / 'evaluation.json')
            assert equal(summary['longitudinal']['changes'][name], comparison(raw[name], parent, 'fresh', reps))
            before = {r['problem_id']: int(r['correct']) for r in parent['sets']['fresh']}
            after = {r['problem_id']: int(r['correct']) for r in raw[name]['sets']['fresh']}
            gains[name] = [after[key] - before[key] for key in sorted(before)]
        assert equal(summary['longitudinal']['difference_in_differences'],
                     interval(np.array(gains['main']) - np.array(gains['reward_only']), reps))
        assert equal(summary['longitudinal']['old_test_difference'], comparison(raw['main'], raw['reward_only'], 'test', reps))
    site = release / 'docs/site' / (slug if slug != 'pilot' else '') / 'status.json'
    state = read(site)
    assert state['summary'] == summary
    for name, rows in state['curves'].items():
        assert rows == read(directory / name / 'metrics.json')
    return dict(summary_sha256=sha256(release / 'releases' / slug / 'summary.json'),
                site_sha256=sha256(site), estimates_intervals_examples_resources_match=True,
                primary=summary['primary'])


def check_history(pilot, continuation, verification):
    """Compare server merge times, raw run clocks and surviving recovery evidence."""
    torch.set_num_threads(1)
    prs = {p['number']: p for p in read(Path('reports/independent-github-chronology.json'))['records']}
    p, c = (read(path / 'experiment.json') for path in (pilot, continuation))
    parent = read(continuation / 'restart-parent/experiment.json')
    checks = {}
    for number, start, label in ((1, p['started_at'], 'initial publication before pilot'),
                                  (7, p['phases']['main']['started_at'], 'weighting amendment before training'),
                                  (19, parent['started_at'], 'continuation registration before execution'),
                                  (20, parent['started_at'], 'continuation core before execution'),
                                  (21, c['started_at'], 'recovery registration before preparation')):
        published = datetime.fromisoformat(prs[number]['merged_at'].replace('Z', '+00:00')).timestamp()
        assert published < start, label
        checks[label] = dict(publication=prs[number], execution_start_utc=datetime.fromtimestamp(start, timezone.utc).isoformat(),
                             seconds_before=round(start - published, 6))
    for relative, expected in c['recovery']['files'].items():
        assert verification['input_files']['continuation'][f'restart-parent/{relative}'] == expected
    before, after = (torch.load(continuation / path, weights_only=True, map_location='cpu', mmap=True)
                     for path in ('restart-parent/main/checkpoint.pt', 'main/checkpoint.pt'))
    assert before['master'].keys() == after['master'].keys()
    assert all(torch.equal(value, after['master'][key]) for key, value in before['master'].items())
    failed = read(pilot / 'failure-evaluation-uncommitted.json')['sets']['sampled']
    final = read(pilot / 'original/evaluation.json')['sets']['sampled']
    assert all({k: v for k, v in a.items() if k != 'batch_seconds'} == {k: v for k, v in b.items() if k != 'batch_seconds'}
               for a, b in zip(failed, final, strict=False))
    assert len(final) >= len(failed)
    return dict(chronology=checks, recovery_source_hashes_match=True, recovered_main_weights_unchanged=True,
                failed_write_prefix_rows=len(failed), replayed_batch_rows=p['calibration']['batch_size'],
                failed_write_replay_matches=True,
                gaps=['Initial recovery control tensors were overwritten; exact initial preservation remains unverified.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('pilot-dir', 'continuation-dir', 'release-dir', 'verification', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    separate_output(args.output.parent, [args.pilot_dir, args.continuation_dir])
    verified = read(args.verification)
    result = {name: check_study(directory, args.release_dir, verified['studies'][name])
              for name, directory in (('pilot', args.pilot_dir), ('continuation', args.continuation_dir))}
    result['history'] = check_history(args.pilot_dir, args.continuation_dir, verified)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
