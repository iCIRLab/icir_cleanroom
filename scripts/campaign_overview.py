#!/usr/bin/env python3
"""Completion counts and termination reasons for a seed campaign.

Reads progress.json and every runs.csv under the results tree; writes its
own output folder and never touches the results.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

NORMAL = 'no relative improvement in 10 consecutive HRS attempts'


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def table(header, rows):
    line = '| ' + ' | '.join(header) + ' |'
    rule = '|' + '|'.join('---' for _ in header) + '|'
    return '\n'.join([line, rule] + ['| ' + ' | '.join(r) + ' |' for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    state = json.loads((args.results/'progress.json').read_text())
    methods = state.get('methods') or sorted(
        {t['method'] for t in state['tasks']})
    counts = Counter(t['status'] for t in state['tasks'])

    rows, missing = [], []
    for task in state['tasks']:
        folder = args.results/f"seed_{task['seed']}"/task['method']/'hrs'
        found = sorted(folder.glob('*/runs.csv')) if folder.is_dir() else []
        if not found:
            missing.append((task['seed'], task['method'], task['status']))
            continue
        for record in read_csv(found[-1]):
            rows.append(dict(record, seed=task['seed'], method=task['method'],
                             status=task['status'],
                             classification=task.get('classification', '')))

    reasons = Counter(r['termination_reason'] for r in rows)
    by_method = {m: Counter(r['termination_reason'] for r in rows
                            if r['method'] == m) for m in methods}
    normal = sum(1 for r in rows if r['termination_reason'] == NORMAL)
    failures = [r for r in rows
                if 'final approach stopped:' in r['termination_reason']]

    parts = [f'# 캠페인 집계: `{args.results.name}`', '',
             f"- 상태: {state.get('status')}",
             f"- 시드: {state['start_seed']}~{state['end_seed']}",
             f"- 방법: {', '.join(methods)}",
             f"- 코드 버전: {(state.get('code_version') or {}).get('describe', '없음')}",
             f"- 전체 태스크: {len(state['tasks'])}건", '',
             '## 완료 건수', '']
    parts.append(table(['상태', '건수'],
                       [[k, str(v)] for k, v in sorted(counts.items())]))
    parts += ['', f'- runs.csv가 있는 실행: {len(rows)}건',
              f'- runs.csv가 없는 태스크: {len(missing)}건', '',
              '## 종료 사유별 건수', '']
    parts.append(table(['종료 사유', '건수', '비율'],
                       [[reason, str(count),
                         f'{count/len(rows):.0%}' if rows else '—']
                        for reason, count in reasons.most_common()]))
    parts += ['', f'- 정상 종료(무개선 10회): **{normal}/{len(rows)}**',
              f'- 최종 접근 실패: {len(failures)}건', '',
              '## 방법별 종료 사유', '']
    all_reasons = list(reasons)
    parts.append(table(['방법'] + [r[:40] for r in all_reasons],
                       [[m] + [str(by_method[m].get(r, 0)) for r in all_reasons]
                        for m in methods]))
    if missing:
        parts += ['', '## runs.csv 없는 태스크', '']
        parts.append(table(['시드', '방법', '상태'],
                           [[str(s), m, st] for s, m, st in missing]))
    if failures:
        parts += ['', '## 최종 접근 실패 상세', '']
        parts.append(table(['시드', '방법', '오차 (m)', '종료 사유'],
                           [[str(r['seed']), r['method'],
                             r.get('source_position_error', '')[:6],
                             r['termination_reason'][:70]] for r in failures]))

    args.output.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(parts) + '\n'
    (args.output/'overview.md').write_text(text, encoding='utf-8')
    (args.output/'overview.json').write_text(json.dumps(dict(
        status=state.get('status'), methods=list(methods),
        start_seed=state['start_seed'], end_seed=state['end_seed'],
        code_version=state.get('code_version'),
        task_counts=dict(counts), runs_with_results=len(rows),
        tasks_without_results=[dict(seed=s, method=m, status=st)
                               for s, m, st in missing],
        termination_reasons=dict(reasons),
        normal_terminations=normal,
        final_approach_failures=len(failures),
    ), indent=2, ensure_ascii=False), encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
