import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
from build_pretrain_v3 import clean
from train_pretrain_v3 import PackedDataset

class DataTests(unittest.TestCase):
    def test_python_indentation_preserved(self):
        code='def f(x):\r\n    if x > 0:\r\n        return x + 1\r\n    return 0\r\n# '+('example '*10)
        result=clean(code)
        self.assertIn('\n    if x',result)
        self.assertIn('\n        return',result)
        compile(result,'sample','exec')

    def test_bad_text_rejected(self):
        self.assertEqual(clean('abc\x00'*100),'')
        self.assertEqual(clean('abc\ufffd'*100),'')

    def test_packed_tail_masked_no_loss_of_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ids=np.arange(2051,dtype='<u4');ids.tofile(root/'train.bin')
            (root/'manifest.json').write_text(json.dumps(dict(status='complete',pad_token_id=151643,outputs={'train':dict(tokens=len(ids),binary={'file':'train.bin'})})))
            ds=PackedDataset(root);self.assertEqual(len(ds),3)
            self.assertTrue(np.array_equal(ds[0]['input_ids'],ids[:1024]))
            self.assertEqual(ds[2]['attention_mask'].sum(),3)
            self.assertTrue((ds[2]['labels'][3:]==-100).all())
            self.assertEqual(sum(int(ds[i]['attention_mask'].sum()) for i in range(len(ds))),len(ids))
            ds.tokens._mmap.close()

if __name__=='__main__':unittest.main()
