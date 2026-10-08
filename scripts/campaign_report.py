#!/usr/bin/env python3
"""Run the whole reporting pipeline over a finished (or partial) campaign.

Each stage is reported separately so a later stage failing still leaves the
earlier output in place, and the run log says which stage stopped.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def stage(name, command, log):
    started = datetime.now(timezone.utc)
    print(f'--- {name}', flush=True)
    done = subprocess.run([sys.executable, *command], capture_output=True,
                          text=True)
    entry = dict(stage=name, command=[str(c) for c in command],
                 returncode=done.returncode,
                 started_utc=started.isoformat(),
                 seconds=(datetime.now(timezone.utc) - started).total_seconds(),
                 stdout_tail=done.stdout[-4000:], stderr_tail=done.stderr[-4000:])
    log.append(entry)
    if done.returncode != 0:
        print(f'    FAILED rc={done.returncode}\n{done.stderr[-2000:]}', flush=True)
    else:
        print(f"    ok ({entry['seconds']:.1f}s)", flush=True)
    return done.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--methods', default='M1,M2,M5,M6')
    parser.add_argument('--reference', default='M1')
    parser.add_argument('--seeds', default='all')
    args = parser.parse_args()
    # The analysis scripts resolve relative paths against the colcon
    # workspace, which is not where results/ lives; pass absolutes.
    args.results = args.results.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    log = []

    ok_overview = stage('overview', [
        SCRIPTS/'campaign_overview.py', '--results', args.results,
        '--output', args.output], log)

    built = args.output/'cohort'
    ok_build = stage('build_cohort', [
        SCRIPTS/'build_cohort.py', '--raw-root', args.results,
        '--methods', args.methods, '--seeds', args.seeds,
        '--output', built], log)

    analysis = args.output/'analysis'
    ok_analyze = ok_build and stage('analyze_cohort', [
        SCRIPTS/'analyze_cohort.py', '--runs', built/'runs_evaluated.csv',
        '--raw-root', args.results, '--methods', args.methods,
        '--reference', args.reference, '--seeds', args.seeds,
        '--output', analysis], log)

    ok_summary = ok_analyze and stage('write_cohort_summary', [
        SCRIPTS/'write_cohort_summary.py',
        '--report', analysis/'report.json',
        '--output', args.output/'summary.md'], log)

    # Record completeness on one finished run, as a spot check.
    sample = next((p.parent for p in sorted(args.results.glob(
        'seed_*/M*/hrs/*/runs.csv'))), None)
    ok_records = sample is not None and stage('check_run_records', [
        SCRIPTS/'check_run_records.py', '--run-dir', sample.parent.parent,
        '--output', args.output/'record_check.md'], log)

    report = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        results=str(args.results), output=str(args.output),
        methods=args.methods, reference=args.reference, seeds=args.seeds,
        stages={e['stage']: e['returncode'] for e in log},
        all_stages_succeeded=all(e['returncode'] == 0 for e in log),
        sample_run=str(sample.parent.parent) if sample else None,
        log=log)
    (args.output/'pipeline.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'log'},
                     indent=2, ensure_ascii=False))
    return 0 if report['all_stages_succeeded'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
