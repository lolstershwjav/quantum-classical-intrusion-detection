# Methodology

## Questions

Do fidelity kernels improve held-out ranking compared with classical kernels on the same low-dimensional input? Do variational classifiers preserve performance under finite-shot readout, and how do sample size and encoding affect the answer? We test these hypotheses on 380 seeded measurements over two training budgets, two feature dimensions, seven models, and four quantum readout conditions.

## Data and matched comparisons

Use 16-dimensional [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) flow measurements that exclude labels, addresses, IDs and history-derived counts. Remove non-finite values, exact feature duplicates and all test vectors equal to vectors from the original training partition. The dataset is a controlled cyber-range release. Training draws of 192 or 384 rows are nested within each seed, with separate 192-row validation and 384-row test sets. Balanced labels support matched comparisons. They inflate the AP reference relative to many deployment class distributions.

Fit log1p, standardization, PCA and rotation scaling using the selected training sample. StandardScaler and PCA depend on training rows, while log1p is fixed. Encode four or six principal components as rotation angles in approximately [−2.25, 2.25]. Give the same transformed rows to each model. Fit linear and RBF SVMs with a common three-value C sweep and ideal validation AP selection. Fit the parameterized circuit, hybrid readout and a small classical MLP with the same seeds, epochs, validation selection and minibatch protocol. Parameter counts differ and are reported.

## Circuits and shot model

The simulator represents an n-qubit pure state with 2ⁿ complex amplitudes. Gates act by indexed bit flips, phases or pair permutations with explicitly little-endian bit order. The angle feature map applies one RY layer. The phase feature map applies RY, a commuting layer of single Z and neighboring ZZ phases, and another RY layer. Fusing the commuting terms into one vectorized pass reduces simulator work without changing the state. A unit test compares this fusion with the unfused gate sequence.

The overlap kernel is K(x,y) = |〈φ(x),φ(y)〉|². To emulate a quantum overlap test, use a Bernoulli probability equal to the fidelity. Draw symmetric binomial estimates for the training matrix. Keep the positive eigenspace of the training Gram matrix and project the validation and test cross-kernels into that space. Test labels and test Gram eigensystems are excluded from the repair.

The variational circuit repeats two blocks of feature-dependent RY, trainable RZ and a directed CNOT ring. A pure VQC reads one Z expectation. The hybrid concatenates all Z expectations with the input angles and fits a classical linear head. Fit their parameters with exact-state differentiation and evaluate readout with exact expectations or computational-basis multinomial samples. Measurements for all Z observables share the same shot outcomes per circuit execution. The 8% noise model mixes final outcome probabilities with a uniform distribution before sampling. This simulates global depolarization and does not model gate-specific errors, connectivity, or physical device time.

For 4 and 6 wires, native-angle feature depth is one. The phase encoding uses depth five under even-ring parallel scheduling and native RZZ gates, so inverse overlap circuits use depth ten before readout. Two trainable blocks have depths 12 and 16 when ring CNOTs follow their implemented sequential order. Physical transpilation may add gates and depth. Simulator timings exclude external device overhead and do not estimate hardware speed.

## Metrics and interpretation

Report average precision, AUROC, balanced accuracy, recall, false-positive rate, fit and inference wall time, negative kernel eigenvalues and parameter or support-vector counts. The validation-normal quantile sets a nominal 5% alert cutoff in each condition. Rerun the kernel SVM under each finite-shot condition. Variational models are fitted once using ideal probabilities and then evaluated under readout changes. The seed bootstrap is paired for quantum-minus-RBF AP. It reflects these samples on one dataset. Data biases, preprocessing, limited dimensions and the simplified noise model constrain generalization.

## References

- Havlíček et al. [Supervised learning with quantum-enhanced feature spaces](https://www.nature.com/articles/s41586-019-0980-2). Nature, 2019.
- Schuld and Killoran. [Quantum Machine Learning in Feature Hilbert Spaces](https://doi.org/10.1103/PhysRevLett.122.040504). Physical Review Letters, 2019.
- Huang et al. [Power of data in quantum machine learning](https://www.nature.com/articles/s41467-021-22539-9). Nature Communications, 2021.
The UNSW dataset terms request these five publications for academic use.

1. Moustafa and Slay. [UNSW-NB15: a comprehensive data set for network intrusion detection systems](https://doi.org/10.1109/MilCIS.2015.7348942), MilCIS 2015.
2. Moustafa and Slay. [The evaluation of Network Anomaly Detection Systems](https://doi.org/10.1080/19393555.2015.1125974), Information Security Journal 2016.
3. Moustafa et al. [Novel geometric area analysis technique for anomaly detection](https://ieeexplore.ieee.org/document/7948715), IEEE Transactions on Big Data 2017.
4. Moustafa et al. [Big data analytics for intrusion detection system using finite Dirichlet mixture models](https://link.springer.com/chapter/10.1007/978-3-319-59439-2_5), 2017.
5. Sarhan et al. [NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems](https://arxiv.org/abs/2011.09144), BDTA 2020.
