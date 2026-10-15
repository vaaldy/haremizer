from pathlib import Path
import json
import tempfile
import unittest

from PIL import Image

from matrix import build, discover_stages, read_log


def write_png(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (768, 768), color).save(path)


class MatrixTests(unittest.TestCase):
    def test_paging_repeats_original_column_and_orders_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / 'run'
            (run / 'baseline-primary').mkdir(parents=True)
            (run / 'baseline-challenge').mkdir(parents=True)
            for i in range(1, 5):
                write_png(run / 'baseline-primary' / f'case-{i:02}.png', 'white')
                write_png(run / 'baseline-challenge' / f'case-{i:02}.png', 'white')
            for number in (1, 2, 3):
                for suite, tone in (('sample-0', 'red'), ('sample-1', 'blue')):
                    for i in range(1, 5):
                        write_png(run / 'loops' / 'loop-01' / f'stage-{number:03}' / suite / f'case-{i:02}.png', tone)
            (run / 'loops' / 'loop-01-preflight-longtokens').mkdir(parents=True)
            (run / 'log.jsonl').write_text('')
            stages = discover_stages(run)
            self.assertEqual([s['index'] for s in stages], [1, 2, 3])
            legend = build(run, Path(folder) / 'out', columns=3, cell=40)
            self.assertEqual(len(legend['pages']), 2)
            self.assertEqual(legend['rows'], 8)
            for page in legend['pages']:
                self.assertTrue(Path(page).is_file())
    def test_read_log_tolerates_concatenated_objects(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'log.jsonl'
            path.write_text('{"index": 1, "score": 0.1}{"index": 2, "score": 0.2}\n')
            entries = read_log(path)
            self.assertEqual(sorted(entries), [1, 2])
            self.assertEqual(entries[2]['score'], 0.2)


if __name__ == '__main__':
    unittest.main()
