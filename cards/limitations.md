# Limitations

- Prepared datasets, final quality measurements, and throughput measurements do
  not exist yet.
- W&B private verification and GitHub authentication are manual external gates.
- BFCL uses a separate Python 3.12 environment because its NumPy pin conflicts
  with the project Python 3.14 environment.
- The fixture notebook is not evidence of real model performance.
- The official IID and Schema-OOD test result is locked and test-once. Exact
  reruns are replicas, not additional official evidence.
- BFCL is a post-selection external check. It cannot select a winner or tune a
  model.
