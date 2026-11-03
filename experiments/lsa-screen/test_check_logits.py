import array
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CheckTest(unittest.TestCase):
    def test_requires_finite_equal_logits_and_all_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b, trace, dense_log, all_log, out = (root / name for name in ("a", "b", "trace", "dense.log", "all.log", "out"))
            meta = "model=/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf input_sha256=209249faab8ffb4bfe2e966b85b9caaa6628a427da237e0e158a76f31fb495ef tokens=45 vocab=3 configured_type_k=bf16 configured_type_v=bf16 ctx=1024 threads=4 threads_batch=4 affinity=0,1,2,3 configured_gpu_layers=0"
            dense_log.write_text('LSA_LOGITS arm=dense ' + meta + '\n')
            all_log.write_text('LSA_LOGITS arm=all ' + meta + '\n')
            def write(path, values):
                path.write_bytes(array.array("f", values).tobytes())
            def run(affinity='0,1,2,3'):
                return subprocess.run([sys.executable, str(Path(__file__).with_name("check_logits.py")),
                                       str(a), str(b), str(trace), str(dense_log), str(all_log), affinity, str(out)], capture_output=True).returncode
            write(a, [0.1, 0.3, -0.5])
            write(b, [0.1, 0.3, -0.5])
            records = [dict(layer=i, queries=45, changed=0, allowed_before=1035, allowed_after=1035) for i in range(3, 64, 4)]
            trace.write_text(''.join(json.dumps(row) + '\n' for row in records))
            self.assertEqual(run(), 0)
            all_log.write_text('LSA_LOGITS arm=all ' + meta.replace('affinity=0,1,2,3', 'affinity=24,25,26,27') + '\n')
            self.assertNotEqual(run(), 0)
            dense_log.write_text('LSA_LOGITS arm=dense ' + meta.replace('affinity=0,1,2,3', 'affinity=24,25,26,27') + '\n')
            self.assertEqual(run('24,25,26,27'), 0)
            self.assertNotEqual(run('1,2,3,4'), 0)
            dense_log.write_text('LSA_LOGITS arm=dense ' + meta + '\n')
            all_log.write_text('LSA_LOGITS arm=all ' + meta + '\n')
            write(a, [0.1, 0.3])
            write(b, [0.1, 0.3])
            self.assertNotEqual(run(), 0)
            write(a, [0.1, 0.3, -0.5])
            write(b, [0.1, 0.301, -0.5])
            self.assertNotEqual(run(), 0)
            write(b, [0.1, float('nan'), -0.5])
            self.assertNotEqual(run(), 0)
            write(b, [0.1, 0.3, -0.5])
            trace.write_text(''.join(json.dumps(row) + '\n' for row in records[:-1]))
            self.assertNotEqual(run(), 0)
            records[0]['changed'] = 1
            trace.write_text(''.join(json.dumps(row) + '\n' for row in records))
            self.assertNotEqual(run(), 0)


if __name__ == '__main__':
    unittest.main()
