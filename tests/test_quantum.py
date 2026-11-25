"""Verify state, gradient, noise and experiment-matching invariants."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from qids.benchmark import project_gram, run
from qids.circuit import Circuit, QuantumNet, fidelity, measure_shots, noisy_overlap
from qids.data import FEATURES, subsets


class QuantumTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_bell_normalization_and_measurement(self):
        c = Circuit(2)
        x = torch.tensor([[np.pi/2,0.]])
        state = c.zero(x)
        rotated = c.ry(state,x[:,0],0)
        bell = rotated[:,c.controls[0]]
        torch.testing.assert_close(bell.abs().square(),torch.tensor([[.5,0,0,.5]]))
        torch.testing.assert_close(c.measure(bell), torch.zeros((1,2)))

    def test_fused_phase_encoding_matches_explicit_gate_sequence(self):
        c = Circuit(4)
        x = torch.tensor(np.random.default_rng(5).normal(size=(4,4)),dtype=torch.float32)
        state = c.ry_layer(c.zero(x),x)
        for q in range(c.qubits):
            state = c.rz(state,-.9*x[:,q],q)
            j = (q+1)%c.qubits
            state = state*torch.exp(.25j*x[:,q:q+1]*x[:,j:j+1]*c.z[:,q]*c.z[:,j])
        state = c.ry_layer(state,.6*x)
        torch.testing.assert_close(state,c.encode(x,"phase"),atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(state.abs().square().sum(1),torch.ones(4))

    def test_quantum_gradient_against_centered_difference(self):
        torch.manual_seed(1)
        net=QuantumNet(4,hybrid=True)
        x=torch.full((3,4),.21)
        out=net(x).sum()
        out.backward()
        index=(0,2,0)
        exact=net.angles.grad[index].item()
        with torch.no_grad():
            old=net.angles[index].item()
            net.angles[index]=old+.001
            plus=net(x).sum().item()
            net.angles[index]=old-.001
            minus=net(x).sum().item()
        self.assertAlmostEqual(exact,(plus-minus)/.002,places=3)

    def test_shots_gram_and_readout(self):
        c=Circuit(4)
        x=torch.full((9,4),.5)
        encode=c.encode(x,"angle").numpy()
        matrix=fidelity(encode,encode)
        rng=np.random.default_rng(6)
        gram=noisy_overlap(matrix,4,.12,256,rng,symmetric=True)
        np.testing.assert_array_equal(gram,gram.T)
        corrected,cross,basis,count=project_gram(gram,gram[:3])
        self.assertGreaterEqual(np.linalg.eigvalsh(corrected).min(),-1e-8)
        self.assertTrue(np.isfinite(cross).all())
        self.assertEqual(len(basis),9)
        self.assertGreaterEqual(count,0)
        np.testing.assert_allclose(measure_shots(encode,c.z.numpy(),0,1,rng),0.,atol=1e-8)

    def test_shared_validation_and_test_sets(self):
        data=self.data()
        first=subsets(data,6,20,20,24)
        second=subsets(data,6,40,20,24)
        for a,b in zip(first[2:],second[2:],strict=True):
            np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(first[0][:10], second[0][:10])

    def data(self):
        rng=np.random.default_rng(1)

        def rows():
            y=np.tile([0,1],70)
            x=np.abs(rng.lognormal(size=(len(y),16))+y[:,None]*.5)
            frame=pd.DataFrame(x,columns=FEATURES)
            frame["label"]=y
            return frame
        return rows(),rows()

    def test_full_small_runtime(self):
        config=dict(seeds=[11],qubits=[4],train_sizes=[60],validation_size=36,
                    test_size=50,epochs=2,conditions=[dict(name="ideal",shots=0,mix=0.),
                                                     dict(name="finite",shots=128,mix=.1)])
        with patch("qids.benchmark.load",return_value=self.data()):
            result=run(Path("."),config)
        self.assertEqual(len(result["runs"]),11)
        json.dumps(result,allow_nan=False)


if __name__=="__main__":
    unittest.main()
