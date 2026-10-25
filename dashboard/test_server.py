import json
import tempfile
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from server import Handler, RunStore


class DashboardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'runs'
        self.run = self.root / 'demo'
        self.run.mkdir(parents=True)
        (self.run / 'train_config.json').write_text(json.dumps({'steps': 10}))
        (self.run / 'metrics.csv').write_text(
            'step,lr,train_loss,val_loss,val_bpc,tokens_per_sec,elapsed_secs\n'
            '1,0.01,4,,,20,1\n2,0.01,,3,4,,2\n3,0.01,NaN,,,inf,3\n4,0.01,2')
        self.store = RunStore(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_live_reads_partial_rows_and_nonfinite_values(self):
        data = self.store.snapshot('demo')
        self.assertEqual(len(data['metrics']), 3)
        self.assertIsNone(data['metrics'][-1]['train_loss'])
        self.assertIsNone(data['metrics'][-1]['tokens_per_sec'])
        self.assertEqual(data['status']['phase'], 'untracked')
        with (self.run / 'metrics.csv').open('a') as file:
            file.write(',,,30,4\n')
        self.assertEqual(len(self.store.snapshot('demo')['metrics']), 4)
        json.dumps(data, allow_nan=False)

    def test_run_path_is_confined_and_symlinks_cannot_escape(self):
        for name in ['..', '../other', '/tmp', 'demo/../..', '']:
            with self.assertRaises(FileNotFoundError):
                self.store.path(name)
        (self.root / 'escape').symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(FileNotFoundError):
            self.store.path('escape')

    def test_http_serves_dashboard_and_neural_data(self):
        (self.run / 'neural.json').write_text('{"step":1,"tensors":[]}')
        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, store=self.store))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            for path in ['/', '/style.css', '/graph.js', '/app.js', '/api/runs', '/api/runs/demo', '/api/runs/demo/neural']:
                with urlopen(base + path) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.headers['Cache-Control'], 'no-store')
            with self.assertRaises(HTTPError) as error:
                urlopen(base + '/api/runs/%2e%2e')
            self.assertEqual(error.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
