# test.py
import unittest
from emirhan_quantum.quantum6g import Quantum6G
import numpy as np

class TestQuantum6G:
    def setUp(self):
        self.num_qubits = 2
        self.qnn = Quantum6G(self.num_qubits)

    def test_evaluate_with_different_weights(self):
        weights = np.array([0.5, 0.5])
        x = np.array([1, 0])

        result = self.qnn.evaluate(weights)

        self.assertAlmostEqual(result, 0.5, places=7)

    def test_evaluate_with_even_more_different_weights(self):
        weights = np.array([0.7, 0.3])
        x = np.array([1, 0])

        result = self.qnn.evaluate(weights)

        self.assertAlmostEqual(result, 0.7, places=7)


if __name__ == '__main__':
    unittest.main()

### Test results:

#----------------------------------------------------------------------
#Ran 0 tests in 0.000s
#
#OK