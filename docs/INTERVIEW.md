# Technical interview preparation

1. **Why use the same PCA features?** It controls input information and compression so the comparison focuses on the model and encoding.
2. **What is the quantum kernel here?** It maps features to normalized states and uses squared state overlap as a kernel for a classical SVM.
3. **How could you estimate overlap on hardware?** Prepare one state's feature map, apply the inverse of the other, and measure the probability of the all-zero basis outcome.
4. **Why can shots produce a non-PSD Gram matrix?** Independent sample noise perturbs pairwise similarities. Its symmetric reconstruction need not remain positive semidefinite.
5. **How is kernel repair kept train-only?** Estimate the positive eigenspace from training outcomes and project validation and test cross-rows using that same basis.
6. **What does a Z expectation measure?** It is the difference between the probabilities of measuring zero and one on that wire. Multiple wire expectations can share basis measurements.
7. **How does the hybrid differ from the VQC?** The VQC uses one measured expectation. The hybrid reads every qubit and gives the classical head direct access to the same compressed inputs.
8. **How was numerical accuracy verified?** Compare an analytical Bell state, fused and unfused gate sequences, exact and mixed measurements, and autograd against finite differences.
9. **Why does larger circuit width cost more?** Statevector dimension grows as 2ⁿ and the number of kernel pairs grows quadratically with sample size. Quantum hardware adds its own shot and compilation costs.
10. **Did the study demonstrate quantum advantage?** The ideal phase feature map ranked slightly better in one comparison. The paired interval included a tie, and the noisy result fell below the matched RBF mean. The project evaluated a classical simulation.

## Résumé drafts

- Benchmarked 380 matched quantum and classical intrusion-detection measurements with two qubit counts, two sample budgets, two encodings and finite-shot noise.
- Implemented a differentiable statevector simulator and train-only kernel repair, verifying the phase optimization and gradient math with analytical and numerical tests.
- Measured 72.51% AP for an ideal six-qubit phase kernel versus 71.02% for an RBF SVM, documenting the paired uncertainty, runtime and noise trade-offs.

The circuit and measurement discussion complements prior QS-DES work. For CrowdStrike's research role, explain why same-input controls, negative results and direct numerical verification make the comparison defensible.
