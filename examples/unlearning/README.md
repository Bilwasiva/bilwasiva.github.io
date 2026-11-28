# A reproducible exact-removal example (ridge regression)

Python 3.10+; standard library only. Run from the repository root:

```bash
python examples/unlearning/exact_ridge_deletion.py
python -m unittest discover -s examples/unlearning -v
```

**What it demonstrates:** Under a fixed feature mapping, a fixed positive L2 penalty, a deterministic quadratic objective, and consistent numeric arithmetic, the fit after removing a row's sufficient-statistics contribution matches fitting the remaining rows from scratch up to numerical precision. It checks repeated deletion requests and prevents reinsertion of an already removed ID within this in-memory demo.

**What it does not demonstrate:** No guarantee about LLM unlearning, deep neural networks, sensitive data erasure from backups or logs, cryptographic proof, differential privacy, legal compliance, or durable distributed operations. The toy object deliberately keeps per-record contribution information in memory in order to reverse it; in real deployments that storage itself requires a deletion/retention policy.