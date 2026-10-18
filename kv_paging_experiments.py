"""
OS x AI, post 1: "Your LLM Is Running a Virtual Memory System"

Two SIMULATIONS (no GPU, no real model). Synthetic workloads, fixed seeds.
  A) KV-cache allocation: reserve-max-length vs paged blocks.
  B) Prefix-cache eviction for agent loops: LRU, random, loop-aware, oracle.

Run:  python kv_paging_experiments.py
Needs: numpy, matplotlib
"""
import heapq
import math
import random
from collections import OrderedDict

import numpy as np

# ---------------------------------------------------------------- Part A
def part_a(seed=0, n=20000, max_len=2048, block=16, capacity_tokens=200_000):
    rng = np.random.default_rng(seed)
    prompt = np.clip(rng.lognormal(math.log(500), 0.9, n), 1, max_len - 1)
    out = np.clip(rng.lognormal(math.log(200), 0.9, n), 1, max_len)
    total = np.minimum(prompt + out, max_len).astype(int)  # final live tokens

    # Static: every request reserves max_len contiguous slots up front.
    reserved_static = np.full(n, max_len)
    # Paged: blocks allocated on demand, last block partly empty.
    reserved_paged = np.ceil(total / block).astype(int) * block

    def stats(reserved):
        util = total.sum() / reserved.sum()
        fit = capacity_tokens / reserved.mean()  # avg concurrent requests that fit
        return util, fit

    return {
        "static": stats(reserved_static),
        "paged": stats(reserved_paged),
        "mean_live_tokens": total.mean(),
    }

# ---------------------------------------------------------------- Part B
def make_events(seed, n_sessions=60, horizon=600.0, base_blocks=64):
    """Agent sessions: each step re-reads its whole prefix, then appends new
    blocks (model output + tool result), then waits for a tool of random latency.
    Step times are fixed up front (they do NOT react to cache misses)."""
    rng = random.Random(seed)
    sessions, events = [], []
    for s in range(n_sessions):
        mean_lat = rng.choice([1.0, 5.0, 30.0])  # fast / medium / slow tools
        t = rng.uniform(0, 60)
        length = base_blocks
        steps = rng.randint(10, 20)
        for k in range(steps):
            growth = max(1, int(rng.lognormvariate(math.log(24), 0.6)))
            events.append((t, s, k, length, growth))  # prefix before this step
            length += growth
            t += rng.expovariate(1.0 / mean_lat)
        sessions.append(mean_lat)
    events.sort()
    return events, sessions


def run_policy(events, capacity, policy, seed=0):
    """Session-level prefix cache with leaf (tail) eviction, as in radix-tree
    prefix caches. cached[s] = number of leading blocks of session s in cache."""
    rng = random.Random(seed)
    cached = {}
    last_access = {}
    est_gap = {}  # running mean of observed gaps, per session (loop-aware)
    used = 0
    hit_blocks = reuse_blocks = 0

    # Oracle: next time each session returns (perfect foresight).
    nxt = {}
    by_session = {}
    for i, (t, s, k, L, g) in enumerate(events):
        by_session.setdefault(s, []).append(t)
    pos = {s: 0 for s in by_session}

    def next_return(s, now):
        ts = by_session[s]
        while pos[s] < len(ts) and ts[pos[s]] <= now:
            pos[s] += 1
        return ts[pos[s]] if pos[s] < len(ts) else float("inf")

    for t, s, k, L, g in events:
        # Reuse: longest cached prefix of this session's old context.
        have = min(cached.get(s, 0), L)
        hit_blocks += have
        reuse_blocks += L
        # Update gap estimate (what a real scheduler could observe).
        if s in last_access:
            gap = t - last_access[s]
            est_gap[s] = gap if s not in est_gap else 0.5 * est_gap[s] + 0.5 * gap
        last_access[s] = t

        need = (L + g) - cached.get(s, 0)  # blocks to insert for this session
        while used + need > capacity:
            victims = [v for v, n in cached.items() if n > 0 and v != s]
            if not victims:
                break
            if policy == "lru":
                v = min(victims, key=lambda x: last_access[x])
            elif policy == "random":
                v = rng.choice(victims)
            elif policy == "loop":
                # predicted return = last access + estimated gap (default: 10s)
                v = max(victims, key=lambda x: last_access[x] + est_gap.get(x, 10.0))
            elif policy == "rate":
                # v2: evict the session that returns least often (largest mean gap).
                # Elapsed time is ignored: with memoryless gaps it carries no information.
                v = max(victims, key=lambda x: est_gap.get(x, 10.0))
            elif policy == "rate_idle":
                # v3: expected wait = the larger of the usual gap and how long the
                # session has already been idle (long idle => probably finished).
                v = max(victims, key=lambda x: max(est_gap.get(x, 10.0), t - last_access[x]))
            elif policy == "oracle":
                v = max(victims, key=lambda x: next_return(x, t))
            else:
                raise ValueError(policy)
            cached[v] -= 1
            used -= 1
        room = capacity - used
        add = min(need, room)
        cached[s] = cached.get(s, 0) + add
        used += add
        if cached[s] < 0:
            cached[s] = 0
    return hit_blocks / reuse_blocks


def peak_working_set(events):
    """Blocks if every session that will return again keeps its context."""
    last = {}
    for t, s, k, L, g in events:
        last[s] = t
    cur, peak, live = {}, 0, {}
    for t, s, k, L, g in events:
        live[s] = L + g
        if last[s] == t:
            live.pop(s)
        peak = max(peak, sum(live.values()))
    return peak


def part_b(seeds=(0, 1, 2, 3, 4), fracs=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75)):
    policies = ["lru", "random", "loop", "rate", "rate_idle", "oracle"]
    table = {p: {f: [] for f in fracs} for p in policies}
    for sd in seeds:
        events, _ = make_events(sd)
        ws = peak_working_set(events)
        for f in fracs:
            cap = max(int(ws * f), 200)
            for p in policies:
                table[p][f].append(run_policy(events, cap, p, seed=sd))
    return table, fracs


def classic_loop_demo(n_pages=50, cache=49, reps=200):
    """OSTEP ch.22 looping-sequential workload: LRU/FIFO 0% hit, RAND >0%."""
    trace = list(range(n_pages)) * reps
    def sim(policy):
        rng = random.Random(0)
        c, order, hits = set(), OrderedDict(), 0
        for p in trace:
            if p in c:
                hits += 1
                if policy == "lru":
                    order.move_to_end(p)
            else:
                if len(c) >= cache:
                    v = next(iter(order)) if policy in ("lru", "fifo") else rng.choice(list(c))
                    c.discard(v); order.pop(v, None)
                c.add(p); order[p] = 1
        return hits / len(trace)
    return {p: sim(p) for p in ("fifo", "lru", "random")}


if __name__ == "__main__":
    a = part_a()
    print("PART A  (static vs paged KV allocation)")
    print(f"  mean live tokens per request: {a['mean_live_tokens']:.0f}")
    for k in ("static", "paged"):
        u, fit = a[k]
        print(f"  {k:7s} memory utilization {u*100:5.1f}%   avg concurrent requests per 200k tokens: {fit:6.1f}")
    print(f"  paged / static concurrency: {a['paged'][1]/a['static'][1]:.2f}x")

    print("\nOSTEP looping-sequential demo (50 pages, cache 49)")
    for p, h in classic_loop_demo().items():
        print(f"  {p:7s} hit rate {h*100:5.1f}%")

    table, fracs = part_b()
    print("\nPART B  (prefix-cache hit rate, mean +- std over 5 seeds; cache as fraction of peak working set)")
    print("  frac   " + "".join(f"{p:>16s}" for p in table))
    for f in fracs:
        row = "".join(f"{np.mean(table[p][f])*100:9.1f}+-{np.std(table[p][f])*100:4.1f}" for p in table)
        print(f"  {f:4.2f}  {row}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    styles = {"lru": ("LRU", "#c0504d", "o"), "random": ("Random", "#8c8c8c", "s"),
              "loop": ("Predicted-return (v1)", "#d98f2b", "v"),
              "rate": ("Return-rate-aware (v2)", "#2b6cb0", "^"),
              "rate_idle": ("Rate + idle-time (v3)", "#2f9e6e", "P"),
              "oracle": ("Oracle (known return)", "#1a1a1a", "D")}
    for p, (label, color, m) in styles.items():
        ys = [np.mean(table[p][f]) * 100 for f in fracs]
        ax.plot([f * 100 for f in fracs], ys, label=label, color=color, marker=m, lw=1.8, ms=5)
    ax.set_xlabel("Cache size (% of peak working set)")
    ax.set_ylabel("Prefix blocks served from cache (%)")
    ax.set_title("Agent-loop prefix cache: eviction policy vs hit rate (synthetic)")
    ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figure1_prefix_cache_policies.png")
    print("\nsaved figure1_prefix_cache_policies.png")
