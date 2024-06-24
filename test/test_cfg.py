import os
import random
import unittest

from cfg_dataset.cfg import CFG


class TestCFG(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        test_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(test_dir)
        self.cfg = CFG(os.path.join(root_dir, "configs/cfg/simple4.cfg"))

    def test_sample_det(self):
        true_samples = ['hhhhhhig', 'hhhhhhgii', 'hhhhhhhgi', 'hhhhhiig',
                        'hhhhhigii', 'hhhhhihgi', 'hhhhigiig', 'hhhhigigii']
        samples = self.cfg.sample_det(n=8)
        self.assertEqual(samples, true_samples)

    def test_sample_rand(self):
        strings = self.cfg.sample_rand(n=10)
        self.assertEqual(len(strings), 10)

        for s in strings:
            self.assertTrue(self.cfg.verify(s))

    def test_verify(self):
        strings = self.cfg.sample_rand(n=10)
        fake_strings = []

        for i in range(len(strings)):
            test_sent = list(strings[i])
            test_sent[0] = "g"
            fake_strings.append("".join(test_sent))

        verifications = [self.cfg.verify(s) for s in fake_strings]
        self.assertIn(False, verifications)


if __name__ == "__main__":
    unittest.main()