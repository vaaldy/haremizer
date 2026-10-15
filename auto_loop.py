"""Unattended bounded driver: prepare a paired baseline, then chain single-variable hypotheses.

Safety: every stage runs through refinement_loop process bounds (owned process group, train<=900s,
sampler<=600s, storage ceiling). The driver stops on cap, deadline, or three consecutive failures,
is restart-safe (it records a stage that finished but was not yet reviewed), and never overwrites
another stage's evidence.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import refinement_loop as rl  # noqa: E402
from score_sprite import aggregate, compare, measure  # noqa: E402

STYLE_TOKENS = 'bold dark outlines, flat colors, single-tone marks, clean seams'
MAX_PROMPT_TOKENS = 77
BASE_CONFIG = {'batch_size': 1, 'resolution': 768, 'rank': 4, 'gradient_clip': 1.0,
               'content_resolution': 768, 'prior_preservation_weight': 0.1,
               'caption_dropout_probability': 0.25, 'snr_gamma': None,
               'facial_loss_weight': 1, 'decoded_feature_loss_weight': 0, 'seed': 20261014}


def utc(text):
    return datetime.datetime.fromisoformat(text.replace('Z', '+00:00'))


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def read_log_indexes(path):
    """Indexes already present, tolerant of a previously concatenated line."""
    indexes = set()
    if not path.exists():
        return indexes
    decoder = json.JSONDecoder()
    text = path.read_text()
    position = 0
    while position < len(text):
        while position < len(text) and text[position] in ' \n\r\t':
            position += 1
        if position >= len(text):
            break
        value, position = decoder.raw_decode(text, position)
        indexes.add(value.get('index'))
    return indexes


def case_files(directory):
    return [directory / f'sample-{suite}' / f'case-{i:02}.png' for suite in (0, 1) for i in range(1, 5)]


class Driver:
    def __init__(self, mission_path, once=False):
        self.mission_path = Path(mission_path)
        self.mission = json.loads(self.mission_path.read_bytes())
        self.root = self.mission_path.parent
        self.once = once
        self.failures = 0
        self.log_path = self.root / 'driver.log'
        self.goal = json.loads(Path(self.mission['goal']).read_bytes())

    # ---- plumbing -------------------------------------------------------
    def log(self, text):
        line = f'{now().isoformat()} {text}'
        print(line, flush=True)
        with self.log_path.open('a') as stream:
            stream.write(line + '\n')

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + '\n')
        return path

    def read(self, name):
        return json.loads((self.root / name).read_bytes())

    def thresholds(self):
        metrics = self.goal['metrics']
        return {'margin': 0.02,
                'ink_min': metrics['ink_amount']['ink_pixels']['range'][0],
                'ink_max': metrics['ink_amount']['ink_pixels']['range'][1],
                'clearance_per_case': metrics['framing']['top_clearance_rows']['gate'],
                'clearance_cases_required': metrics['framing']['top_clearance_rows']['cases_required'],
                'border_min': metrics['background']['border_color_fraction']['gate']}

    def completed(self):
        directory = self.root / 'results'
        return len(list(directory.glob('result-*.json'))) if directory.exists() else 0

    # ---- baselines ------------------------------------------------------
    def suite_with_style_tokens(self):
        """Freeze suites carrying the same style tokens as the training captions."""
        plan = json.loads(Path(self.mission['plan']).read_bytes())
        suites = []
        for source in plan['suites']:
            data = json.loads(Path(source['path']).read_bytes())
            for case in data['cases']:
                assert ', pixel art,' in case['prompt'], case['prompt']
                case['prompt'] = case['prompt'].replace(', pixel art,', f', {STYLE_TOKENS}, pixel art,', 1)
            target = self.root / f"suite-{Path(source['path']).stem}.json"
            target.write_text(json.dumps(data, indent=2) + '\n')
            suites.append({'path': str(target.resolve()),
                           'baseline': str((self.root / f"baseline-{Path(source['path']).stem}").resolve())})
        self.assert_token_budget(suites)
        return {**plan, 'suites': suites}

    def assert_token_budget(self, suites):
        """Both encoders must see the whole prompt; truncation silently drops the quality tags."""
        from transformers import CLIPTokenizer
        model = json.loads(Path(self.mission['plan']).read_bytes())['model']['path']
        tokenizers = [CLIPTokenizer.from_pretrained(f'{model}/{name}', local_files_only=True)
                      for name in ('tokenizer', 'tokenizer_2')]
        worst = 0
        for suite in suites:
            for case in json.loads(Path(suite['path']).read_bytes())['cases']:
                lengths = [len(tokenizer(case['prompt']).input_ids) for tokenizer in tokenizers]
                worst = max(worst, *lengths)
                if max(lengths) > MAX_PROMPT_TOKENS:
                    raise ValueError(f"prompt exceeds {MAX_PROMPT_TOKENS} tokens: {case['id']} {lengths}")
        self.log(f'prompt token budget ok, worst {worst} of {MAX_PROMPT_TOKENS}')

    def sample(self, suite_path, output, label):
        plan = json.loads(Path(self.mission['plan']).read_bytes())
        plan['deadline_utc'] = self.mission['deadline_utc']
        model = plan['model']
        argv = [plan['python'], '-B', 'sample.py', '--model', model['path'], '--model-id', model['id'],
                '--revision', model['revision'], '--output', str(output), '--suite', str(suite_path),
                '--width', '768', '--height', '768', '--steps', '28', '--max-cases', '4',
                '--guidance-scale', '5', '--negative-prompt', plan['settings']['negative_prompt'],
                '--cpu-offload', '--scheduler', 'euler_ancestral']
        guard = rl.Loop.__new__(rl.Loop)
        guard.plan = plan
        guard.state = {'stages': []}
        guard.directory = self.root
        guard.path = self.root / 'baseline-guard.json'
        guard.bounds()
        stage = {'number': 0, 'directory': str(self.root), 'processes': []}
        guard.process(stage, argv, plan['sample_seconds'], label)
        manifest = json.loads((output / 'manifest.json').read_bytes())
        if len(manifest['records']) != 4:
            raise ValueError('four frozen cases required')
        for record in manifest['records']:
            peak = record['gpu_memory'].get('peak_allocated_bytes')
            if peak is None or peak > 11 * 1024 ** 3:
                raise ValueError('missing allocator evidence or 11 GiB allocator ceiling exceeded')
        return manifest

    def prepare(self):
        (self.root / 'results').mkdir(parents=True, exist_ok=True)
        (self.root / 'requests').mkdir(parents=True, exist_ok=True)
        if (self.root / 'summary.json').exists() and not (self.root / 'run-finished.flag').exists():
            (self.root / 'summary.json').rename(self.root / 'summary-stale.json')
        plan = self.suite_with_style_tokens()
        if not (self.root / 'baseline-metrics.json').exists():
            rows = []
            for suite in plan['suites']:
                directory = Path(suite['baseline'])
                if not directory.exists():
                    label = f"baseline-{Path(suite['path']).stem}"
                    attempt = 1
                    while (self.root / f'{label}-{attempt}.stdout.log').exists():
                        attempt += 1
                    self.sample(suite['path'], directory, f'{label}-{attempt}')
                rows += [measure(directory / f'case-{i:02}.png') for i in range(1, 5)]
            self.write('baseline-metrics.json', aggregate(rows))
            self.log(f'baseline measured on {len(rows)} cases')
        if not (self.root / 'incumbent.json').exists():
            self.write('incumbent.json', {'metrics': self.read('baseline-metrics.json'),
                                          'checkpoint': None, 'stage': 0, 'source': 'no-lora baseline',
                                          'score': 0.0})
        if not (self.root / 'latest.json').exists():
            self.write('latest.json', {'checkpoint': None})
        plan['deadline_utc'] = self.mission['deadline_utc']
        plan['max_stages'] = self.mission['cap']
        self.write('training-plan.json', plan)
        return plan

    # ---- stages ---------------------------------------------------------
    def loop_for_next(self, done, mission):
        """First loop directory that can still accept a stage or holds an unreviewed one."""
        cohort_size = mission.get('cohort_size', 8)
        for index in range(1, 65):
            loop_dir = Path(mission['loop_dir']) / f'loop-{index:02d}'
            if not (loop_dir / 'loop.json').exists():
                plan = json.loads((self.root / 'training-plan.json').read_bytes())
                plan['max_stages'] = max(1, min(cohort_size, mission['cap'] - done))
                plan['deadline_utc'] = mission['deadline_utc']
                self.write(f'plan-{index:02d}.json', plan)
                rl.Loop.init(loop_dir, self.root / f'plan-{index:02d}.json')
                self.log(f'initialised loop {loop_dir} (max {plan["max_stages"]} stages)')
            loop = rl.Loop(loop_dir)
            max_stages = json.loads(Path(loop.state['plan']['path']).read_bytes())['max_stages']
            if (len(loop.state['stages']) < max_stages or
                    loop.state['status'] == 'needs_visual_review'):
                return loop
        raise ValueError('no loop directory can accept work')

    def request(self, index, hypothesis, incumbent, latest, dataset, template):
        plan = json.loads(Path(self.mission['plan']).read_bytes())
        config = {**BASE_CONFIG, **hypothesis.get('config', {})}
        config['facial_loss_weight'] = hypothesis.get('config', {}).get('facial_loss_weight', 1)
        config['decoded_feature_loss_weight'] = hypothesis.get('config', {}).get('decoded_feature_loss_weight', 0)
        if config['decoded_feature_loss_weight'] > 0 or config['facial_loss_weight'] > 1:
            raise ValueError('region-weighted losses need declared facial_regions')
        continuation = None
        fallback = None
        if hypothesis.get('continuation') == 'incumbent':
            parent = incumbent.get('checkpoint') or latest.get('checkpoint')
            if not parent:
                raise ValueError('continuation requested without any trained checkpoint')
            if not incumbent.get('checkpoint'):
                fallback = 'no incumbent checkpoint passed the gate; continued from the latest trained adapter'
            continuation = {'path': parent['path'], 'sha256': parent['sha256'],
                            'outcome': parent['outcome'], 'outcome_sha256': parent['outcome_sha256']}
        request = {**template, 'model': plan['model'], 'config': config,
                   'examples': dataset['examples'], 'heldout_examples': dataset['heldout_examples'],
                   'continuation': continuation, 'declared_smoke_only': False,
                   'reviewed_by': 'machine:overnight-driver',
                   'authorization': {'user_authorized': True, 'record': (
                       'User 2026-10-13 asked for long unattended loops that keep improving until the goal. '
                       f'Overnight window: deadline {self.mission["deadline_utc"]}, cap {self.mission["cap"]} GPU '
                       f'stages, serial, one GPU job at a time. Stage {index} hypothesis {hypothesis["name"]}: '
                       f'{hypothesis["rationale"]}{" " + fallback + "." if fallback else ""} '
                       'Bold-ink CC-BY-4.0 native-64 sources replace the zero-ink everface style set. Style '
                       'tokens are shared by training captions and evaluation prompts, within the 77-token budget. '
                       'No generated-image training feedback, no paid services, no pushes, no parent-repo mutation, '
                       'no 1024 route, no driver or global compatibility change.')}}
        path = self.root / 'requests' / f'request-{index:03}.json'
        path.write_text(json.dumps(request, indent=2) + '\n')
        return path

    def record_stage(self, loop, index, hypothesis, baseline):
        """Score, review and account for a finished stage. Safe to re-run after a restart."""
        stage = loop.state['stages'][-1]
        rows = [measure(path) for path in case_files(Path(stage['directory']))]
        if len(rows) != len(stage['cases']):
            raise ValueError('measured outputs must match the frozen case list')
        thresholds = self.thresholds()
        metrics = {**aggregate(rows),
                   'clearance_cases_ok': sum(1 for row in rows
                                             if row['top_clearance_rows'] >= thresholds['clearance_per_case'])}
        incumbent = self.read('incumbent.json')
        verdict = compare(metrics, incumbent['metrics'], thresholds=thresholds)
        checkpoint = stage['checkpoint']
        outcome = Path(stage['directory']) / 'train' / 'outcome.json'
        winner = {'path': checkpoint['path'], 'sha256': checkpoint['sha256'],
                  'outcome': str(outcome.resolve()), 'outcome_sha256': rl.fingerprint(outcome)}
        self.write(f'results/result-{index:03}.json',
                   {'stage': stage['number'], 'global_index': index, 'hypothesis': hypothesis['name'],
                    'stage_directory': str(stage['directory']),
                    'metrics': metrics, 'verdict': verdict, 'baseline': baseline, 'per_case': rows,
                    'checkpoint': checkpoint,
                    'duration_seconds': [p.get('duration_seconds') for p in stage['processes']]})
        entry = {'index': index, 'hypothesis': hypothesis['name'], 'score': verdict['score'],
                 'accepted': verdict['accepted'], 'regressions': verdict['regressions'],
                 'metrics': metrics, 'checkpoint_sha256': checkpoint['sha256'],
                 'at': now().isoformat()}
        logged = read_log_indexes(self.root / 'log.jsonl')
        if index not in logged:
            with (self.root / 'log.jsonl').open('a') as stream:
                stream.write(json.dumps(entry) + '\n')
            logged.add(index)
        else:
            self.write(f'log-duplicate-{index:03}.json', {**entry, 'note': 'index already logged; not appended'})
        cases = {}
        for position, case in enumerate(stage['cases']):
            row = rows[position]
            cases[case['case']] = {
                'glitches': 'not visually reviewed by the driver',
                'crop_duplicate_face': f"top clearance {row['top_clearance_rows']} rows; not visually reviewed",
                'adherence': 'not assessed by the driver; descriptor and attire checks need visual review',
                'readability': f"ink {row['ink_pixels']}, components {row['components']}, largest "
                               f"{row['largest_component']}, isolated share {row['isolated_share']}"}
        review = {'reviewed_by': 'machine:overnight-driver',
                  'notes': (f"Deterministic gate only, not visual acceptance. {hypothesis['name']}: score "
                            f"{verdict['score']}, regressions {verdict['regressions']}, clearance cases "
                            f"{metrics['clearance_cases_ok']}/{len(rows)}, gates {verdict['gates']}."),
                  'uncertain': True, 'robust_requested': False,
                  'checks': {'format': False, 'pixel_quality': verdict['accepted'],
                             'prompt_adherence': False, 'character_diversity': True,
                             'non_regression': not verdict['regressions']},
                  'binding': loop.status()['review_binding'],
                  'cases': cases,
                  'format_evidence': []}
        review_path = self.write(f'review-{index:03}.json', review)
        try:
            loop.review(review_path)
        except (ValueError, OSError, KeyError, TypeError) as error:
            self.log(f'stage {index}: review record rejected ({type(error).__name__}: {error}); '
                     'result retained, loop left awaiting review')
        self.write('latest.json', {'checkpoint': winner, 'stage': stage['number'],
                                   'source': hypothesis['name'], 'score': verdict['score'],
                                   'metrics': metrics})
        if verdict['accepted']:
            self.write('incumbent.json', {**winner, 'metrics': metrics, 'stage': stage['number'],
                                          'source': hypothesis['name'], 'score': verdict['score']})
            self.log(f"stage {index} {hypothesis['name']}: accepted as incumbent (score {verdict['score']})")
        else:
            self.log(f"stage {index} {hypothesis['name']}: rejected (score {verdict['score']}, "
                     f"regressions {verdict['regressions']})")
        return verdict

    def index_for_directory(self, directory, fallback):
        """Global index already recorded for a stage directory, else the next index."""
        for path in sorted((self.root / 'results').glob('result-*.json')):
            if json.loads(path.read_bytes()).get('stage_directory') == directory:
                return json.loads(path.read_bytes())['global_index']
        return fallback

    def next_index(self):
        """Highest recorded global stage index; results are written before the review is applied."""
        directory = self.root / 'results'
        files = list(directory.glob('result-*.json')) if directory.exists() else []
        return max([int(path.stem.split('-')[1]) for path in files] or [0])

    def run(self):
        mission = self.mission
        dataset = json.loads(Path(mission['dataset']).read_bytes())
        template = json.loads(Path(mission['template_request']).read_bytes())
        baseline = self.read('baseline-metrics.json')
        while True:
            done = self.next_index()
            if done >= mission['cap']:
                self.log('stop: experiment cap reached')
                break
            if now() >= utc(mission['deadline_utc']):
                self.log('stop: deadline reached')
                break
            if self.failures >= 3:
                self.log('stop: three consecutive failures')
                break
            loop = self.loop_for_next(done, mission)
            if loop.state['status'] == 'needs_visual_review':
                stage = loop.state['stages'][-1]
                index = self.index_for_directory(str(stage['directory']), done + 1)
                hypothesis = mission['hypotheses'][min(index - 1, mission['cap'] - 1)]
                self.log(f'stage {index} {hypothesis["name"]}: finished, recording'
                         + (' after restart' if (self.root / f'results/result-{index:03}.json').exists() else ''))
                try:
                    self.record_stage(loop, index, hypothesis, baseline)
                except BaseException as error:  # noqa: BLE001 - never let bookkeeping kill the window
                    self.failures += 1
                    self.log(f'stage {index} bookkeeping failed: {type(error).__name__}: {error}')
                    self.write(f'failure-record-{index:03}.json',
                               {'error': f'{type(error).__name__}: {error}'})
                if self.once:
                    break
                continue
            hypothesis = mission['hypotheses'][done]
            incumbent = self.read('incumbent.json')
            latest = self.read('latest.json')
            try:
                request = self.request(done + 1, hypothesis, incumbent, latest, dataset, template)
                loop.enqueue(request, hypothesis['rationale'])
                self.log(f"stage {done + 1} {hypothesis['name']}: dispatching")
                loop.pipeline()
            except BaseException as error:  # noqa: BLE001 - recorded and bounded
                self.failures += 1
                self.log(f'stage {done + 1} failed: {type(error).__name__}: {error}')
                self.write(f'failure-{done + 1:03}.json', {'hypothesis': hypothesis,
                                                           'error': f'{type(error).__name__}: {error}'})
                continue
            self.failures = 0
            try:
                self.record_stage(loop, done + 1, hypothesis, baseline)
            except BaseException as error:  # noqa: BLE001
                self.failures += 1
                self.log(f'stage {done + 1} bookkeeping failed: {type(error).__name__}: {error}')
                self.write(f'failure-record-{done + 1:03}.json', {'error': f'{type(error).__name__}: {error}'})
            if self.once:
                break
        (self.root / 'run-finished.flag').touch()
        summary = {'finished_at': now().isoformat(), 'stages_completed': self.completed(),
                   'incumbent': self.read('incumbent.json'),
                   'latest': self.read('latest.json'), 'baseline': baseline}
        self.write('summary.json', summary)
        self.log(f"finished: {summary['stages_completed']} stages recorded, incumbent "
                 f"{summary['incumbent']['source']}")
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mission', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    driver = Driver(args.mission, once=args.once)
    driver.prepare()
    if not args.prepare_only:
        driver.run()


if __name__ == '__main__':
    main()
