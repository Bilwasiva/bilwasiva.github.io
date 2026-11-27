# Educational physical-AI closed-loop simulator

No external dependencies. Requires Python 3.9+.

```bash
python examples/physical_ai/toy_safe_mpc.py
cd examples/physical_ai
python -m unittest -v test_toy_safe_mpc.py
```

This is a minimal 1D deterministic simulation for illustration, not robot control or a proof of safety. The planner estimates dynamics with drag 0.985, while the plant uses drag 0.94. The code intentionally separates a proposed action from an independent, approximate one-step admissibility filter and checks the resulting state after each simulated action. The shield is not robust to arbitrary disturbances, perceptual uncertainty, or sensor failure.