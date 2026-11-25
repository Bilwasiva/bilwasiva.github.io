# The Economics of Thinking — companion code

The script `reasoning_test_time_compute_demo.py` is a **dependency-free educational simulation** of inference-time candidate sampling and verifier-based selection. It is not a benchmark of a real model, and the generated success rates must not be presented as empirical model performance.

Run:

```bash
python3 reasoning_test_time_compute_demo.py --trials 20000 --seed 7
python3 -m unittest -v test_reasoning_demo.py
```

The code intentionally separates oracle pass@N (whether any sampled answer is correct) from the success of the selected candidate. It compares fixed N=1, N=4, N=8, and a hypothetical difficulty-aware allocation. Difficulty classes and score distributions are synthetic, and the adaptation policy is known by construction, not learned.

The blog's canonical Jekyll path is `_posts/2026-11-25-the-economics-of-thinking.md`. Its cover is placed in `assets/img/blog/reasoning-models-test-time-compute-cover.jpg`.