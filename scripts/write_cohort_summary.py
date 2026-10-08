#!/usr/bin/env python3
"""Render a Korean reading guide for one analyse_cohort.py output folder."""
import argparse
import json
from pathlib import Path

VERDICT = {'no_detected_difference': '차이 검출 안 됨',
           'better_than_reference': '기준보다 좋음',
           'worse_than_reference': '기준보다 나쁨',
           'identical_on_every_paired_seed': '모든 시드에서 동일',
           'not_compared': '비교 불가'}
START = {'patrol_peak': '순찰 최고 농도 지점',
         'patrol_centroid': '순찰 농도 가중 무게중심'}
SEARCH = {'estimate_max': '추정 평균 최대 셀', 'spiral': '나선 탐색'}


def fmt(value, digits=1):
    return '없음' if value is None else f'{value:.{digits}f}'


def pct(value):
    return '없음' if value is None else f'{value:.0%}'


def table(header, rows):
    line = '| ' + ' | '.join(header) + ' |'
    rule = '|' + '|'.join('---' for _ in header) + '|'
    return '\n'.join([line, rule] + ['| ' + ' | '.join(r) + ' |' for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    r = json.loads(args.report.read_text())
    out = args.output or args.report.parent/'summary.md'
    methods = r['methods']
    stats = {s['method']: s for s in r['method_stats']}
    stages = {s['method']: s for s in r['stage_summary']}

    parts = [f"# 코호트 분석 요약 ({r['seed_count']}개 시드 × {len(methods)}개 방법 = {r['run_count']}건)",
             '',
             f"- 시드: {r['seeds'][0]}~{r['seeds'][-1]} ({r['seed_count']}개, 전체)",
             f"- 평가 입력: `{r['inputs']['runs']}` (build_cohort.py 산출)",
             f"- 원시 실행: `{r['inputs']['raw_root']}`",
             f"- 기준 방법: {r['reference_method']}",
             f"- 판정 규칙: {r['notes']['verdict_rule']}",
             f"- 절단 처리: {r['notes']['censoring']}",
             '',
             '## 0. 비교 설계 (2×2)',
             '']
    design = r['design']
    starts = list(dict.fromkeys(d['start'] for d in design.values()))
    searches = list(dict.fromkeys(d['search'] for d in design.values()))
    grid = {(d['start'], d['search']): m for m, d in design.items()}
    parts.append(table([''] + [SEARCH.get(x, x) for x in searches],
                       [[START.get(st, st)] + [grid.get((st, se), '-') for se in searches]
                        for st in starts]))
    merged = r.get('merged_methods') or {}
    if merged:
        evidence = {(a['method_a'], a['method_b']): a for a in r['first_cell_agreement']}
        parts.append('')
        for dropped, kept in merged.items():
            key = (kept, dropped) if (kept, dropped) in evidence else (dropped, kept)
            a = evidence.get(key)
            # A campaign that never ran the merged method has no evidence
            # to quote; the merge note is simply omitted.
            if a is None or not a.get('compared') or (
                    a.get('within_one_cell_rate') is None):
                continue
            parts.append(
                f"- {dropped}은 {kept}와 첫 목표 셀이 완전히 같은 경우 "
                f"{a['identical']}/{a['compared']}회, 한 칸 이내가 "
                f"{a['within_one_cell']}/{a['compared']}회 "
                f"({a['within_one_cell_rate']:.0%}), 거리 중앙값 "
                f"{fmt(a['first_cell_distance_median'], 2)} m다. 시작 위치가 사실상 "
                f"같으므로 {kept}에 통합했다.")
    parts += ['',
              '## 1. 방법별 분포',
              '',
              '주 지표는 **결정 시간**이다. 탐색 전환부터 최종 추정이 확정될 때까지이며, '
              '그 뒤의 최종 접근은 3절에서 따로 본다. 전체 시간 T는 결정 시간과 최종 '
              '접근을 합한 보조 지표다. 중앙값과 사분위 범위를 쓴다.',
              '']
    parts.append(table(
        ['방법', 'n', '결정 시간 중앙', '결정 시간 IQR', 'T 중앙', 'T IQR',
         'E 중앙', 'E IQR', '정확 추정', '오차 ≤0.5 m', '최종 접근 실패'],
        [[m, str(stats[m].get('n')),
          fmt(stats[m].get('T_decision_median')), fmt(stats[m].get('T_decision_iqr')),
          fmt(stats[m].get('T_median')), fmt(stats[m].get('T_iqr')),
          fmt(stats[m].get('E_median'), 3), fmt(stats[m].get('E_iqr'), 3),
          f"{stats[m].get('E_exact_hits')}/{stats[m].get('n')}",
          pct(stats[m].get('E_at_or_below_0.5')),
          f"{stats[m].get('navigation_failed')} "
          f"({pct(stats[m].get('navigation_failure_rate'))})"]
         for m in methods]))
    parts += ['',
              '`정확 추정`은 최종 추정 셀이 기준 셀과 일치한 실행 수다. 로봇이 그 셀까지 '
              '실제로 갔는지와는 별개이며, 도달 여부는 3절에서 본다. 오차 분포는 0에 몰려 '
              '있어 평균이 소수의 큰 실패에 끌려가므로 누적 분포와 정확 추정 수를 쓴다.',
              '',
              '## 2. 최종 접근 (주 지표 아님)',
              '',
              '최종 추정이 확정된 뒤 그 셀로 이동한 구간이다. 여기서 실패해도 탐색이 실패한 '
              '것은 아니므로 결정 시간에는 반영하지 않는다.',
              '']
    parts.append(table(
        ['방법', '도착', '도착률', '접근 시간 중앙(도착분)', '접근 시간 IQR'],
        [[m, f"{stats[m].get('approach_succeeded_count', 0)}/{stats[m].get('n')}",
          pct(stats[m].get('approach_success_rate')),
          fmt(stats[m].get('T_approach_arrived_median')),
          fmt(stats[m].get('T_approach_arrived_iqr'))]
         for m in methods]))
    design_contrasts = r.get('paired_design_contrasts') or []
    if design_contrasts:
        parts += ['', '## 2b. 2x2 설계 대비 (주 결과)', '',
                  '설계가 정의한 네 쌍만 비교하고, 홈 보정도 이 네 쌍에 대해서만 한다. '
                  '음수는 왼쪽 방법이 더 좋다는 뜻이다. 결정 시간에는 절단을 적용하지 '
                  '않는다. 최종 접근에 실패한 실행도 추정 자체는 측정된 시각에 확정했기 '
                  '때문이다.', '']
        rows = []
        for entry in design_contrasts:
            ci = (f"[{fmt(entry['median_ci_low'], 2)}, {fmt(entry['median_ci_high'], 2)}]"
                  if entry.get('median_ci_low') is not None else '—')
            rows.append([f"{entry['method']} vs {entry['reference']}", entry['metric'],
                         f"{entry['wins']}/{entry['ties']}/{entry['losses']}",
                         pct(entry.get('win_rate')),
                         fmt(entry.get('diff_median'), 2), ci,
                         fmt(entry.get('wilcoxon_p'), 4),
                         fmt(entry.get('wilcoxon_p_holm'), 4),
                         VERDICT.get(entry['verdict'], entry['verdict'])])
        parts.append(table(['대비', '지표', '승/무/패', '승률', '차이 중앙',
                            '중앙 95% CI', 'Wilcoxon p', 'Holm p', '판정'], rows))
    parts += ['',
              f"## 2c. 기준 방법({r['reference_method']}) 대비 시드별 차이 (보조)",
              '',
              '같은 시드에서 각 방법과 기준의 차이. 음수는 기준보다 좋음. 시간은 주행 실패를 '
              '패배로 처리하고(절단), 크기 통계는 양쪽 다 도착한 쌍에서만 계산한다. 오차는 '
              '실패 실행도 유효하므로 그대로 쓴다. p값은 윌콕슨 부호순위이며 지표별 홈 보정이다. '
              '대부분의 오차 쌍이 0에서 동률이라 판정 방향은 승패 균형으로 정한다.',
              '']
    rows = []
    for entry in r['paired_vs_reference']:
        ci = (f"[{fmt(entry['median_ci_low'], 2)}, {fmt(entry['median_ci_high'], 2)}]"
              if entry.get('median_ci_low') is not None else '—')
        rows.append([entry['method'], entry['metric'],
                     f"{entry['wins']}/{entry['ties']}/{entry['losses']}",
                     pct(entry.get('win_rate')),
                     fmt(entry.get('diff_median'), 2), ci, str(entry['censored_pairs']),
                     fmt(entry.get('wilcoxon_p'), 4), fmt(entry.get('wilcoxon_p_holm'), 4),
                     VERDICT.get(entry['verdict'], entry['verdict'])])
    parts.append(table(['방법', '지표', '승/무/패', '승률', '차이 중앙', '중앙 95% CI',
                        '절단 쌍', 'Wilcoxon p', 'Holm p', '판정'], rows))
    parts += ['',
              '## 3. 단계별 지표',
              '',
              '첫 목표 셀과 가스원의 거리, 전환부터 첫 셀 측정까지, 첫 셀부터 최고 농도 최초 '
              '측정까지, 그 뒤 종료까지. 모두 중앙값 [IQR]. 마지막 열은 도착에 성공한 실행만 센다.',
              '']
    parts.append(table(['방법', '첫 셀–가스원 (m)', '→ 첫 측정 (s)', '첫 측정 → 최고 (s)',
                        '최고 → 종료 (s, 도착분)'],
                       [[m,
                         f"{fmt(stages[m].get('first_target_source_distance_m_median'), 2)} [{fmt(stages[m].get('first_target_source_distance_m_iqr'), 2)}]",
                         f"{fmt(stages[m].get('time_to_first_measurement_median'))} [{fmt(stages[m].get('time_to_first_measurement_iqr'))}]",
                         f"{fmt(stages[m].get('time_first_measurement_to_best_median'))} [{fmt(stages[m].get('time_first_measurement_to_best_iqr'))}]",
                         f"{fmt(stages[m].get('time_best_to_end_median'))} [{fmt(stages[m].get('time_best_to_end_iqr'))}]"]
                        for m in methods]))
    parts += ['', '## 4. 첫 목표 셀 일치율 (통합 근거 포함)', '']
    parts.append(table(['방법 쌍', '비교', '완전 일치', '한 칸 이내', '거리 중앙값 (m)'],
                       [[f"{a['method_a']}–{a['method_b']}", str(a['compared']),
                         f"{a['identical']} ({pct(a.get('identical_rate'))})",
                         f"{a['within_one_cell']} "
                         f"({pct(a.get('within_one_cell_rate'))})",
                         fmt(a.get('first_cell_distance_median'), 2)]
                        for a in r['first_cell_agreement']]))
    parts += ['', '## 5. 종합 점수와 가중치 민감도', '',
              '보조 요약이다. 주 결과는 2절의 짝지은 비교다.', '']
    scores = r['weighted_scores']
    parts.append(table(['가중치 (시간:오차)'] + methods,
                       [[label] + [(f"{scores[label]['score'][m]:.2f} "
                                    f"({scores[label]['rank'][m]}위)")
                                   if scores[label]['score'].get(m) is not None
                                   else '없음'
                                   for m in methods] for label in scores]))
    changed = r['rank_stability']['changed_methods']
    parts += ['',
              '6:4 순위: ' + ' > '.join(r['rank_stability']['orderings'][
                  r['rank_stability']['reference_weighting']]),
              '']
    for label, same in r['rank_stability']['identical_to_reference'].items():
        parts.append(f"- {label} 가중치: 순위 {'유지' if same else '변동'}"
                     + ('' if same else f" (바뀐 방법: {', '.join(changed[label])})"))
    parts += ['', '## 6. 주행 완료율', '',
              '최종 접근 실패는 탐색 실패가 아니다. 실패한 실행의 오차를 함께 적어 둔다.', '']
    parts.append(table(['방법', '실행', '정상', '최종 접근 실패', '실패율', '실패 중 오차 0',
                        '실패 시드'],
                       [[c['method'], str(c['runs']), str(c['normal']),
                         str(c['final_approach_failed']),
                         pct(c.get('final_approach_failed_rate')),
                         str(c['failed_with_zero_error']),
                         ', '.join(str(x) for x in c['failed_seeds']) or '—']
                        for c in r['completion_rates']]))
    parts.append('')
    parts += ['## 7. 조건별 결과', '',
              f"순찰 최근접 거리 중앙값 {r['condition_cuts']['nearest_patrol_distance_median']:.2f} m를 경계로 나눴다. "
              'σ와 강도는 console.log에서 복구한 값이다.', '']
    current = None
    for b in r['condition_bins']:
        if (b['condition'], b['bin']) != current:
            current = (b['condition'], b['bin'])
            parts += ['', f"**{b['condition']} — {b['bin']}**", '']
            rows = []
        rows.append([b['method'], str(b['n']), fmt(b.get('T_median')), fmt(b.get('T_iqr')),
                     fmt(b.get('E_median'), 3), fmt(b.get('E_mean'), 3)])
        if b['method'] == r['methods'][-1]:
            parts.append(table(['방법', 'n', 'T 중앙', 'T IQR', 'E 중앙', 'E 평균'], rows))
    parts += ['', '## 8. 생성 파일', '']
    for name in sorted(p.name for p in args.report.parent.iterdir()
                       if p.suffix in ('.csv', '.png')):
        parts.append(f'- `{name}`')
    parts += ['', '## 9. 주의', '',
              f"- 판정: {r['notes']['verdict_rule']}",
              f"- 절단: {r['notes']['censoring']}",
              f"- 설계: {r['notes']['design']}",
              f"- 보고: {r['notes']['reporting']}",
              f"- {r['notes']['source_truth']}",
              f"- σ·강도 복구: {r['source_truth_recovery']['recovered']}/{r['source_truth_recovery']['total']} 시드, "
              f"좌표 일치 {r['source_truth_recovery']['coordinate_matches']}건",
              '- 원본 결과 파일은 읽기만 했고 수정하지 않았다.', '']
    Path(out).write_text('\n'.join(parts), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
