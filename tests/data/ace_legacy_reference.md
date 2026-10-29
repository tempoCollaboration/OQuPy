# ACE legacy numerical reference

The adjacent ace_legacy_reference.npz was generated from the supplied
ACE_OQuPy_Integration source on 2026-09-14.

- ProcessTensor.py SHA-256:
  c982550359e9e50262bd13c03de4f97bda4bf4b444186a313fccb2d15726483f
- MPO_Compression.py SHA-256:
  a6b56db3f24eb08007005e91d23bdb42b735e324350f7af85f472a4a554c94a6

The original Process_Tensor_mapping, Process_Tensor and MPO_Compression class
definitions were evaluated with the original Operators and PhysicalConstants.
Only exact_correlation_function was replaced with a no-op to prevent diagnostic
plotting and hardcoded file writes; it does not feed the process tensor.
Propagator construction, bath initialization, influence tensors and
preselection/compression math were left unchanged.

Inputs: omega=[0.7,-0.4,1.3], g=[0.21,0.14,0.09], levels=[2,3,2],
S=diag(0,1), dt=0.125, four steps, Temp=150 (ignored by this mapping class),
tol=1e-10 and method="Preselection".
The source requires at least three modes for this path to avoid its
single/two-mode early-return bugs.

The legacy result was converted with import_ace_process_tensor and contracted
with the stored Hamiltonian and initial state using OQuPy's existing dynamics.
The resulting full density matrices are stored as expected in the fixture.
The new builder differs by at most 1.0672e-9 on this case at the same local
threshold; compression gauges and merge order are allowed to differ.
The regression tolerance is 1e-8. Independent dense joint-system tests in
ace_test.py separately validate the converter and dynamics conventions.
