#!/usr/bin/env python3
"""Check that a finished run folder records everything an analysis needs.

One row per required item, so a missing column is visible before a campaign
runs rather than after. Reads only; never writes into the run folder.
"""
import argparse
import csv
import json
from pathlib import Path

ITEMS = (
    ('identity', '시드'),
    ('identity', '방법'),
    ('identity', '코드 버전'),
    ('identity', '설정'),
    ('source', '가스원 위치'),
    ('source', '가스원 분포 폭'),
    ('source', '가스원 강도'),
    ('patrol', '순찰 지점별 위치'),
    ('patrol', '순찰 지점별 측정값'),
    ('patrol', '순찰 소요 시간'),
    ('search', '목표 셀 좌표'),
    ('search', '목표 셀 선택 시각'),
    ('search', '측정 위치'),
    ('search', '측정값'),
    ('search', '측정 시각'),
    ('estimate', '최종 추정 위치'),
    ('estimate', '추정 확정 시각'),
    ('approach', '최종 접근 도착 시각'),
    ('approach', '최종 접근 도착 위치'),
    ('approach', '최종 접근 성공 여부'),
    ('distance', '이동 거리'),
    ('termination', '종료 사유'),
    ('termination', '주행 실패'),
    ('termination', '실패한 목표 셀'),
)


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def present(row, *names):
    return bool(row) and all(
        row.get(name) not in (None, '') for name in names)


def event_rows(events, kind):
    out = []
    for row in events:
        if row.get('event') != kind:
            continue
        try:
            row = dict(row, _details=json.loads(row.get('details') or '{}'))
        except ValueError:
            row = dict(row, _details={})
        out.append(row)
    return out


def detail(rows, *names):
    return bool(rows) and all(
        rows[0]['_details'].get(name) is not None for name in names)


def inspect(method_dir):
    """Collect evidence for one <seed>/<method> folder."""
    hrs = sorted(method_dir.glob('hrs/*'))
    log_dir = hrs[-1] if hrs else method_dir / 'hrs'
    runs = read_csv(log_dir / 'runs.csv')
    events = read_csv(log_dir / 'events.csv')
    lrs_runs = read_csv(log_dir / 'lrs_runs.csv')
    patrol = read_csv(log_dir / 'lrs_measurements.csv')
    run = runs[-1] if runs else {}
    manifest_path = method_dir.parent / 'manifest.json'
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.is_file() else {})
    config = method_dir / 'run_config.yaml'
    selected = event_rows(events, 'target_selected')
    measured = event_rows(events, 'measurement_complete')
    confirmed = event_rows(events, 'final_estimate_selected')
    approach_end = event_rows(events, 'final_approach_end')
    nav_failed = event_rows(events, 'navigation_failed')
    version = manifest.get('code_version') or {}

    return {
        '시드': (manifest.get('seed') is not None, f"manifest.json seed={manifest.get('seed')}"),
        '방법': (present(run, 'method_id'), f"runs.csv method_id={run.get('method_id')}"),
        '코드 버전': (bool(version.get('commit')),
                   f"manifest.json code_version.commit={str(version.get('commit'))[:12]}"
                   + (' (dirty)' if version.get('dirty') else '')),
        '설정': (config.is_file() and (method_dir.parent / 'snapshot').is_dir(),
               'run_config.yaml + snapshot/ + runs.csv parameters'),
        '가스원 위치': (present(run, 'source_x', 'source_y'),
                   f"runs.csv source_x/y=({run.get('source_x')},{run.get('source_y')})"),
        '가스원 분포 폭': (present(run, 'source_sigma'),
                     f"runs.csv source_sigma={run.get('source_sigma')}"),
        '가스원 강도': (present(run, 'source_strength'),
                   f"runs.csv source_strength={run.get('source_strength')}"),
        '순찰 지점별 위치': (bool(patrol) and present(patrol[0], 'robot_x', 'robot_y'),
                      f'lrs_measurements.csv robot_x/y ({len(patrol)} rows)'),
        '순찰 지점별 측정값': (bool(patrol) and present(patrol[0], 'concentration'),
                       'lrs_measurements.csv concentration, hazard'),
        '순찰 소요 시간': (bool(patrol) and present(patrol[0], 'lrs_elapsed_seconds')
                     and bool(lrs_runs) and present(lrs_runs[0], 'lrs_seconds'),
                     'lrs_measurements.csv lrs_elapsed_seconds + lrs_runs.csv lrs_seconds'),
        '목표 셀 좌표': (detail(selected, 'x', 'y', 'variable'),
                    f'events.csv target_selected x/y/variable ({len(selected)} events)'),
        '목표 셀 선택 시각': (bool(selected) and present(selected[0], 'ros_seconds'),
                       'events.csv target_selected ros_seconds'),
        '측정 위치': (detail(measured, 'xy'),
                  f'events.csv measurement_complete xy ({len(measured)} events)'),
        '측정값': (detail(measured, 'value'), 'events.csv measurement_complete value'),
        '측정 시각': (bool(measured) and present(measured[0], 'ros_seconds'),
                  'events.csv measurement_complete ros_seconds'),
        '최종 추정 위치': (present(run, 'estimated_source_x', 'estimated_source_y'),
                     f"runs.csv estimated_source_x/y=({run.get('estimated_source_x')},"
                     f"{run.get('estimated_source_y')})"),
        '추정 확정 시각': (bool(confirmed) and present(confirmed[0], 'ros_seconds'),
                     'events.csv final_estimate_selected ros_seconds'),
        '최종 접근 도착 시각': (bool(approach_end) and present(approach_end[0], 'ros_seconds'),
                        'events.csv final_approach_end ros_seconds'),
        '최종 접근 도착 위치': (detail(approach_end, 'robot_x', 'robot_y'),
                        'events.csv final_approach_end robot_x/y + runs.csv final_robot_x/y'),
        '최종 접근 성공 여부': (detail(approach_end, 'succeeded'),
                        'events.csv final_approach_end succeeded'
                        + (f"={approach_end[0]['_details'].get('succeeded')}"
                           if approach_end else '')),
        '이동 거리': (present(run, 'lrs_distance_m', 'hrs_distance_m', 'total_distance_m'),
                  f"runs.csv total_distance_m={run.get('total_distance_m')}"),
        '종료 사유': (present(run, 'termination_reason'),
                  f"runs.csv termination_reason={run.get('termination_reason')}"),
        '주행 실패': (present(run, 'navigation_attempts'),
                  f"runs.csv navigation_attempts={run.get('navigation_attempts')}, "
                  f'events.csv navigation_failed x{len(nav_failed)}'),
        '실패한 목표 셀': (not nav_failed or detail(nav_failed, 'target_x', 'target_y'),
                     'events.csv navigation_failed target_x/y/variable'
                     + (' (no failures in this run)' if not nav_failed else '')),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='A <results>/seed_N/<method> folder')
    parser.add_argument('--output', type=Path, help='Write the table as Markdown')
    args = parser.parse_args()
    found = inspect(args.run_dir)
    lines = [f'# 기록 점검: `{args.run_dir}`', '',
             '| 구분 | 항목 | 기록 | 위치 |', '|---|---|---|---|']
    missing = []
    for group, name in ITEMS:
        ok, where = found[name]
        lines.append(f"| {group} | {name} | {'O' if ok else 'X'} | {where} |")
        if not ok:
            missing.append(name)
    lines += ['', f"미기록 {len(missing)}건"
              + (': ' + ', '.join(missing) if missing else ' (전 항목 기록됨)')]
    text = '\n'.join(lines)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    raise SystemExit(1 if missing else 0)


if __name__ == '__main__':
    main()
