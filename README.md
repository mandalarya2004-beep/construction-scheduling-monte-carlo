# Resource-Constrained Construction Scheduling Under Uncertainty

**GitHub:** https://github.com/mandalarya2004-beep/construction-scheduling-monte-carlo

Comparing scheduling priority rules with Monte Carlo simulation, benchmarked
against the Critical Path Method (CPM).

## Problem

Construction projects have limited crews/equipment, so the Critical Path Method
(CPM) alone is insufficient — CPM assumes unlimited resources and fixed, certain
durations. Neither assumption holds on a real job site: crews are limited
(Resource-Constrained Project Scheduling, RCPSP), and task durations vary
(weather, labor availability, material delays).

## Approach

1. **CPM** — classical, resource-unconstrained, deterministic baseline
   (theoretical minimum duration).
2. **Resource-constrained priority-rule heuristics** — simple, realistic
   scheduling rules: shortest-task-first, most-successors-first, and
   most-resource-intensive-first.
3. **Monte Carlo simulation** — each priority rule is run 3,000 times with task
   durations resampled from a triangular distribution each run, producing a full
   makespan distribution (not just a single point estimate) per rule.
4. **Sensitivity analysis** — Spearman rank correlation between each task's
   sampled duration and the resulting project makespan, to identify which of the
   17 tasks actually drive schedule risk.
5. **What-if capacity check** — re-runs the Monte Carlo comparison with one crew
   unit removed, to see how each rule's risk profile shifts under a tighter
   resource constraint.

This mirrors how scheduling risk is actually communicated on real projects:
contractors care about "what's the chance we finish by day X," not just a single
point estimate.

## Key results

| Metric | Value |
|---|---|
| CPM lower bound (unconstrained) | 43 days |
| Best deterministic heuristic (most-resource-first) | 46 days |
| Mean makespan under uncertainty, best rule | ~49.3 days |
| Spread between best/worst rule (nominal capacity) | 2.13 days |
| Spread between best/worst rule (1 crew unit lost) | 0.00 days |

**Headline finding:** losing a crew unit makes the choice of priority rule
matter *less*, not more — under a tighter resource constraint, every rule ends
up bottlenecked at the same capacity ceiling. This tells a planner that
restoring crew capacity is likely to help the schedule more than re-optimizing
task order.

The sensitivity analysis identifies **Framing (ground + upper floor)**,
**Interior finishes**, and **Foundation cure** as the tasks whose duration
uncertainty contributes most to overall schedule risk — the tasks most worth
prioritizing for buffer, reliable crews, or early procurement.

## Limitations

- Task durations are sampled **independently** per task via triangular
  distributions; in reality, delays are often correlated (e.g. a bad weather
  week affects several outdoor tasks simultaneously). Correlated sampling would
  be a natural and more realistic extension.
- Only one pooled resource type (crew units) is modeled; separating labor vs.
  equipment constraints would be more realistic for a real site.
- The project network (17 tasks) is illustrative rather than benchmarked
  against a published RCPSP instance (e.g. PSPLIB) — see Extensions below.

## Setup

```bash
pip install -r requirements.txt
python scheduling_simulation.py
```

Running as a Jupyter notebook instead: this script uses the `# %%` cell-marker
format, recognized directly by VS Code and Jupyter (via `jupytext`) as notebook
cells — open it directly in either, or run
`jupytext --to notebook scheduling_simulation.py` to generate a `.ipynb` file.

## Possible extensions

- Correlated duration sampling (e.g. a shared weather-delay factor across
  outdoor tasks)
- Multiple resource types (labor vs. equipment) instead of one pooled resource
- Cost-based objective (cost of delay vs. cost of adding a crew) instead of
  pure makespan minimization
- Benchmark against a PSPLIB instance to compare against published
  heuristic/optimal results

## Project structure

```
.
├── scheduling_simulation.py   # full pipeline: CPM -> heuristics -> Monte Carlo -> sensitivity
└── requirements.txt
```
