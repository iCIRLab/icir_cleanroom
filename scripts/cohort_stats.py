#!/usr/bin/env python3
"""Distribution, paired-difference, and composite-score statistics."""
from collections import Counter
import math
import random
import statistics

from icir_cleanroom.gas_mapping.application.benchmark import cost, score

# The 1 SD rule was retired: the arrival-time tails make it reject every real effect.
VERDICT_RULE = ('paired per seed; Wilcoxon signed-rank with Holm correction '
                'within each metric, alpha = 0.05')


def quantile(values, fraction):
    """Linear-interpolation quantile; no SciPy dependency."""
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def describe(values, label):
    """Mean, median, quartiles, and spread for one metric."""
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return {f'{label}_n': 0}
    q1, q3 = quantile(numbers, .25), quantile(numbers, .75)
    return {
        f'{label}_n': len(numbers),
        f'{label}_mean': statistics.fmean(numbers),
        f'{label}_median': statistics.median(numbers),
        f'{label}_q1': q1, f'{label}_q3': q3, f'{label}_iqr': q3 - q1,
        f'{label}_sd': statistics.stdev(numbers) if len(numbers) > 1 else 0.,
        f'{label}_min': min(numbers), f'{label}_max': max(numbers),
    }


def sign_test(diffs, tolerance=1e-9):
    """Two-sided exact binomial test on the signs; ties are dropped."""
    negative = sum(d < -tolerance for d in diffs)
    positive = sum(d > tolerance for d in diffs)
    trials = negative + positive
    if trials == 0:
        return dict(sign_n=0, sign_negative=0, sign_positive=0, sign_p=None)
    smaller = min(negative, positive)
    tail = sum(math.comb(trials, k) for k in range(smaller + 1)) / 2 ** trials
    return dict(sign_n=trials, sign_negative=negative, sign_positive=positive,
                sign_p=min(1., 2 * tail))


def bootstrap_interval(values, estimator=statistics.median, draws=4000, seed=12345):
    """Percentile bootstrap interval; the tails here are too heavy for normal theory."""
    numbers = [float(v) for v in values if v is not None]
    if len(numbers) < 2:
        return dict(ci_low=None, ci_high=None, ci_draws=0)
    rng = random.Random(seed)
    size = len(numbers)
    estimates = sorted(estimator([numbers[rng.randrange(size)] for _ in range(size)])
                       for _ in range(draws))
    return dict(ci_low=estimates[int(.025 * draws)],
                ci_high=estimates[min(draws - 1, int(.975 * draws))],
                ci_draws=draws)


def threshold_table(values, thresholds, tolerance=1e-9):
    """Cumulative fraction at or below each accuracy requirement."""
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return {}
    return {f'at_or_below_{t:g}': sum(v <= t + tolerance for v in numbers) / len(numbers)
            for t in thresholds}


def signed_rank(diffs, tolerance=1e-9):
    """Wilcoxon signed-rank with average ranks for ties and a normal approximation.

    Zero differences are dropped (Wilcoxon's own convention). With no non-zero
    difference the test has no sample, which is itself the finding.
    """
    values = [d for d in diffs if abs(d) > tolerance]
    n = len(values)
    if n == 0:
        return dict(wilcoxon_n=0, wilcoxon_w=None, wilcoxon_z=None, wilcoxon_p=None)
    ordered = sorted(range(n), key=lambda i: abs(values[i]))
    ranks = [0.] * n
    index = 0
    while index < n:
        stop = index
        while stop + 1 < n and math.isclose(abs(values[ordered[stop + 1]]),
                                            abs(values[ordered[index]]), rel_tol=1e-12):
            stop += 1
        average = (index + stop) / 2. + 1.
        for position in range(index, stop + 1):
            ranks[ordered[position]] = average
        index = stop + 1
    positive = math.fsum(r for r, v in zip(ranks, values) if v > 0)
    negative = math.fsum(r for r, v in zip(ranks, values) if v < 0)
    statistic = min(positive, negative)
    mean = n * (n + 1) / 4.
    tie_groups = Counter(round(abs(v), 12) for v in values).values()
    correction = math.fsum(t ** 3 - t for t in tie_groups) / 48.
    variance = n * (n + 1) * (2 * n + 1) / 24. - correction
    if variance <= 0:
        return dict(wilcoxon_n=n, wilcoxon_w=statistic, wilcoxon_z=None, wilcoxon_p=None)
    # Continuity correction, matching scipy's default for the normal approximation.
    z = (statistic - mean + .5) / math.sqrt(variance)
    p = 2 * (.5 * math.erfc(-z / math.sqrt(2)))
    return dict(wilcoxon_n=n, wilcoxon_w=statistic, wilcoxon_z=z,
                wilcoxon_p=min(1., max(0., p)))


def holm_adjust(entries, key='wilcoxon_p', target='wilcoxon_p_holm', family='metric'):
    """Holm-Bonferroni within each metric family; the comparisons share a reference."""
    for name in {e.get(family) for e in entries}:
        group = [e for e in entries if e.get(family) == name and e.get(key) is not None]
        order = sorted(group, key=lambda e: e[key])
        running = 0.
        for index, entry in enumerate(order):
            adjusted = min(1., (len(order) - index) * entry[key])
            running = max(running, adjusted)  # enforce monotonicity
            entry[target] = running
        for entry in entries:
            if entry.get(family) == name and entry.get(key) is None:
                entry[target] = None
    return entries


def compare_pair(by_key, seeds, method, reference, metric, censoring):
    """One method against one baseline on one metric, seed by seed."""
    outcomes, diffs, paired = [], [], 0
    for seed in seeds:
        left, right = by_key.get((seed, method)), by_key.get((seed, reference))
        if left is None or right is None:
            continue
        if left.get(metric) is None or right.get(metric) is None:
            continue
        paired += 1
        left_bad = censoring and bool(left.get('censored'))
        right_bad = censoring and bool(right.get('censored'))
        if left_bad and right_bad:
            outcomes.append(0)
        elif left_bad:
            outcomes.append(1)
        elif right_bad:
            outcomes.append(-1)
        else:
            difference = float(left[metric]) - float(right[metric])
            diffs.append(difference)
            outcomes.append(0 if difference == 0 else
                            (-1 if difference < 0 else 1))
    wins = outcomes.count(-1)
    ties = outcomes.count(0)
    losses = outcomes.count(1)
    entry = dict(method=method, reference=reference, metric=metric,
                 censoring_applied=censoring, paired_n=paired,
                 wins=wins, ties=ties, losses=losses,
                 win_rate=wins / paired if paired else None,
                 win_rate_excluding_ties=(wins / (wins + losses)
                                          if wins + losses else None),
                 censored_pairs=paired - len(diffs),
                 **describe(diffs, 'diff'))
    entry.update(sign_test([float(o) for o in outcomes]))
    entry.update(signed_rank(diffs))
    entry.update({'median_' + k: v for k, v in
                  bootstrap_interval(diffs).items()})
    return entry


def paired_comparisons(rows, pairs, metrics, censored_metrics=('T',)):
    """Per-seed differences for an explicit list of (method, baseline) pairs.

    Holm runs over the whole family reported here, so the correction matches
    the comparisons actually made rather than every method against one
    reference.
    """
    by_key = {(r['seed'], r['method']): r for r in rows}
    seeds = sorted({r['seed'] for r in rows})
    results = [compare_pair(by_key, seeds, method, reference, metric,
                            metric in censored_metrics)
               for method, reference in pairs for metric in metrics]
    holm_adjust(results)
    return _verdicts(results)


def paired_differences(rows, reference, methods, metrics, censored_metrics=('T',)):
    """Per-seed differences against one reference method.

    A run whose final approach failed has no meaningful arrival time - its value
    records when the retry budget ran out. For such metrics the pair is scored as
    a loss (censoring), and the magnitude statistics are reported on the
    uncensored subset only, with both counts kept side by side.
    """
    by_key = {(r['seed'], r['method']): r for r in rows}
    seeds = sorted({r['seed'] for r in rows})
    results = [compare_pair(by_key, seeds, method, reference, metric,
                            metric in censored_metrics)
               for method in methods if method != reference
               for metric in metrics]
    holm_adjust(results)
    return _verdicts(results)


def _verdicts(results):
    for entry in results:
        adjusted = entry.get('wilcoxon_p_holm')
        low, high = entry.get('median_ci_low'), entry.get('median_ci_high')
        if entry.get('wilcoxon_n') == 0:
            entry['verdict'] = 'identical_on_every_paired_seed'
        elif adjusted is None:
            entry['verdict'] = 'not_compared'
        elif adjusted >= .05:
            entry['verdict'] = 'no_detected_difference'
        else:
            # Most pairs tie at zero, so the median carries no sign; use the
            # win/loss balance, falling back to the median only when it is level.
            balance = entry['wins'] - entry['losses']
            improved = balance > 0 or (balance == 0 and (entry.get('diff_median') or 0) < 0)
            entry['verdict'] = ('better_than_reference' if improved
                                else 'worse_than_reference')
        entry['median_ci_excludes_zero'] = (
            None if low is None else bool(low > 0 or high < 0))
    return results


def weighted_scores(rows, methods, weights=(.6, .5, .4)):
    """Composite score per time weight, so weight sensitivity is visible."""
    table = {}
    for weight in weights:
        label = f'{int(weight*10)}{int(round((1-weight)*10))}'
        values = {}
        for method in methods:
            costs = [cost(float(r['T']), float(r['E']), weight)
                     for r in rows if r['method'] == method
                     and r.get('T') is not None and r.get('E') is not None]
            values[method] = score(costs) if costs else None
        ranks = rank(values)
        table[label] = dict(weight_time=weight, weight_error=round(1 - weight, 10),
                            score=values, rank=ranks)
    return table


def rank(values):
    """Rank 1 = highest score; ties share the better rank."""
    present = {k: v for k, v in values.items() if v is not None}
    return {k: (1 + sum(other > v + 1e-10 for other in present.values())
                if v is not None else None)
            for k, v in values.items()}


def rank_stability(table):
    """Report whether the 6:4 ordering survives 5:5 and 4:6."""
    labels = list(table)
    base = table[labels[0]]['rank']
    return dict(
        reference_weighting=labels[0],
        orderings={label: sorted(table[label]['rank'],
                                 key=lambda m: (table[label]['rank'][m] or 99, m))
                   for label in labels},
        identical_to_reference={label: table[label]['rank'] == base
                                for label in labels[1:]},
        changed_methods={label: sorted(m for m in base
                                       if table[label]['rank'][m] != base[m])
                         for label in labels[1:]})
