#!/usr/bin/env python3
"""Figures for the cohort report. Labels are English so any system font renders."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical order (dataviz reference palette, slots 1-6).
SERIES = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300')
# Secondary encoding: the palette's CVD margin is at the 8-10 band, and these
# figures must also survive grayscale printing.
DASHES = ((), (5, 2), (1, 1.6), (7, 2, 1.5, 2), (3, 1.5, 1.5, 1.5), (9, 3))
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_MUTED = '#52514e'
GRID = '#e3e2dd'


def style(axes, title, xlabel, ylabel):
    axes.set_facecolor(SURFACE)
    axes.set_title(title, color=INK, fontsize=11, loc='left', pad=8)
    axes.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    axes.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    axes.grid(True, color=GRID, linewidth=.6, zorder=0)
    axes.set_axisbelow(True)
    for side in ('top', 'right'):
        axes.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=INK_MUTED, labelsize=8.5)


def palette(methods):
    return {m: SERIES[i % len(SERIES)] for i, m in enumerate(methods)},  \
           {m: DASHES[i % len(DASHES)] for i, m in enumerate(methods)}


def error_cdf(rows, methods, path, cap=None, floor=.7):
    """One step line per method, zoomed to where the runs actually differ."""
    colors, dashes = palette(methods)
    figure, axes = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    limit = cap or max(r['E'] for r in rows if r['E'] is not None)
    exact = {}
    for method in methods:
        values = sorted(r['E'] for r in rows if r['method'] == method
                        and r['E'] is not None)
        if not values:
            continue
        exact[method] = sum(v <= 1e-9 for v in values) / len(values)
        xs, ys = [0.], [0.]
        for index, value in enumerate(values):
            xs += [value, value]
            ys += [index / len(values), (index + 1) / len(values)]
        xs.append(limit)
        ys.append(1.)
        axes.step(xs, ys, where='post', color=colors[method], linewidth=2.,
                  dashes=dashes[method], zorder=3, label=method)
    style(axes, 'Corrected position error, cumulative distribution',
          'Error threshold (m)', 'Fraction of runs at or below')
    axes.set_xlim(-.08, limit * 1.03)
    axes.set_ylim(floor, 1.005)
    lowest = min(exact.values()) if exact else floor
    axes.annotate(
        f'Runs reaching the reference cell exactly: {lowest:.0%}-{max(exact.values()):.0%}\n'
        f'(the step at 0 m); the axis starts at {floor:.0%}, not 0',
        xy=(.03, floor + (1 - floor) * .02), color=INK_MUTED, fontsize=8.5)
    axes.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED,
                loc='lower right', ncol=3)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return path


def decision_time_cdf(rows, methods, path):
    """Decision time is the primary metric, so it gets its own distribution."""
    colors, dashes = palette(methods)
    figure, axes = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    present = [r['T_decision'] for r in rows if r.get('T_decision') is not None]
    if not present:
        style(axes, 'Decision time, cumulative distribution',
              'Decision time (s)', 'Fraction of runs at or below')
        axes.annotate('No decision time recorded in these runs',
                      xy=(.5, .5), xycoords='axes fraction', ha='center',
                      color=INK_MUTED, fontsize=10)
        figure.tight_layout()
        figure.savefig(path, facecolor=SURFACE)
        plt.close(figure)
        return path
    limit = max(present)
    medians = {}
    for method in methods:
        values = sorted(r['T_decision'] for r in rows if r['method'] == method
                        and r.get('T_decision') is not None)
        if not values:
            continue
        medians[method] = values[len(values)//2]
        xs, ys = [0.], [0.]
        for index, value in enumerate(values):
            xs += [value, value]
            ys += [index/len(values), (index + 1)/len(values)]
        xs.append(limit)
        ys.append(1.)
        axes.step(xs, ys, where='post', color=colors[method], linewidth=2.,
                  dashes=dashes[method], zorder=3, label=method)
    style(axes, 'Decision time (search start to final estimate), '
                'cumulative distribution',
          'Decision time (s)', 'Fraction of runs at or below')
    axes.set_xlim(0, limit*1.03)
    axes.set_ylim(0, 1.005)
    if medians:
        axes.annotate(
            'Medians: ' + ', '.join(f'{m} {v:.0f}s' for m, v in medians.items()),
            xy=(.03, .04), xycoords='axes fraction',
            color=INK_MUTED, fontsize=8.5)
    axes.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED,
                loc='lower right', ncol=3)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return path


def time_error_panels(rows, methods, path, metric='T_decision',
                      axis_label='Decision time (s), log scale'):
    """Small multiples: a scatter needs all-pairs colour separation, so facet."""
    colors, _ = palette(methods)
    columns = 3
    panel_rows = (len(methods) + columns - 1) // columns
    figure, grid = plt.subplots(panel_rows, columns, figsize=(9.6, 3.2 * panel_rows),
                                dpi=200, sharex=True, sharey=True)
    figure.patch.set_facecolor(SURFACE)
    flat = list(grid.ravel()) if hasattr(grid, 'ravel') else [grid]
    for index, (axes, method) in enumerate(zip(flat, methods)):
        subset = [r for r in rows if r['method'] == method
                  and r.get(metric) is not None and r['E'] is not None]
        axes.scatter([r[metric] for r in subset], [r['E'] for r in subset],
                     s=28, color=colors[method], alpha=.75,
                     edgecolor=SURFACE, linewidth=1., zorder=3)
        bottom_row = index >= len(methods) - columns
        first_column = index % columns == 0
        style(axes, method, axis_label if bottom_row else '',
              'Corrected error E (m)' if first_column else '')
        axes.set_xscale('log')
        hits = sum(r['E'] <= 1e-9 for r in subset)
        axes.annotate(f'{hits}/{len(subset)} at E = 0',
                      xy=(.97, .93), xycoords='axes fraction', ha='right',
                      color=INK_MUTED, fontsize=8.5)
    for axes in flat[len(methods):]:
        axes.set_visible(False)
    figure.suptitle(f'{axis_label.split(" (")[0]} against corrected error, '
                    'one panel per method',
                    color=INK, fontsize=11.5, x=.012, ha='left')
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return path


def stage_decomposition(stages, methods, path):
    """Median seconds per stage; medians do not sum, so the total is annotated."""
    fields = (('time_to_first_measurement', 'HRS start to first cell'),
              ('time_first_measurement_to_best', 'First cell to best measurement'),
              ('time_best_to_end', 'Best measurement to stop'))
    by_method = {s['method']: s for s in stages}
    figure, axes = plt.subplots(figsize=(7.6, 4.2), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    positions = range(len(methods))
    bottom = [0.] * len(methods)
    for index, (field, label) in enumerate(fields):
        values = [by_method[m].get(field + '_median') or 0. for m in methods]
        axes.bar(positions, values, bottom=bottom, width=.62,
                 color=SERIES[index], label=label, zorder=3,
                 edgecolor=SURFACE, linewidth=2.)  # 2px surface gap between segments
        for position, value, base in zip(positions, values, bottom):
            if value > 12:
                axes.text(position, base + value / 2, f'{value:.0f}',
                          ha='center', va='center', color='#ffffff', fontsize=8.5)
        bottom = [b + v for b, v in zip(bottom, values)]
    style(axes, 'Median seconds per stage (medians are not additive)',
          '', 'Seconds')
    axes.set_xticks(list(positions))
    axes.set_xticklabels(methods)
    axes.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED, loc='upper right')
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return path


def read_runs(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ('T', 'T_decision', 'T_approach', 'E', 'E_raw', 'D'):
            row[key] = float(row[key]) if row.get(key) not in (None, '') else None
    return rows
