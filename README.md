# os-meets-ai

Experiments for the blog series that connects *Operating Systems: Three Easy Pieces* to LLM serving, agents and generative AI.

## Post 1: Your LLM Is Running a Virtual Memory System

`kv_paging_experiments.py` runs two **simulations** (synthetic workloads, fixed seeds, no GPU):

- Static vs paged KV-cache allocation.
- Prefix-cache eviction for agent loops: LRU, random, three heuristics, and an oracle.

```
pip install numpy matplotlib
python kv_paging_experiments.py
```

It prints the tables and writes `figure1_prefix_cache_policies.png`. Output from one run is in `results.txt` (about 50 seconds on a laptop CPU).

These are simulations, not benchmarks of vLLM or any real model. Real-trace and GPU replications are planned follow-ups.
