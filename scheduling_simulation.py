# %% [markdown]
# # Resource-Constrained Construction Scheduling Under Uncertainty
#
# Comparing scheduling priority rules with Monte Carlo simulation, benchmarked
# against the Critical Path Method.
#
# **Problem:** Construction projects have limited crews/equipment, so the Critical
# Path Method (CPM) alone is insufficient — CPM assumes unlimited resources and
# fixed, certain durations. Neither assumption holds on a real job site: crews are
# limited (Resource-Constrained Project Scheduling, RCPSP), and task durations vary
# (weather, labor availability, material delays).
#
# **Approach:**
# 1. **CPM** — classical, resource-unconstrained, deterministic baseline
#    (theoretical minimum duration)
# 2. **Resource-constrained priority-rule heuristics** — simple, realistic
#    scheduling rules (e.g. shortest task first, most-successors first)
# 3. **Monte Carlo simulation** — run each priority rule thousands of times with
#    randomly sampled task durations, to get not just a single makespan estimate
#    but a full distribution: expected duration, and the probability of finishing
#    by a target date
#
# This mirrors how scheduling risk is actually communicated on real projects —
# contractors care about "what's the chance we finish by day X," not just a single
# point estimate.

# %% [markdown]
# ## 1. Setup and imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(42)
plt.rcParams['figure.figsize'] = (10, 5)

# %% [markdown]
# ## 2. Define the project: tasks, durations, precedence, and resources
#
# Each task has a most-likely duration, a resource requirement (crew units), and
# predecessors. We will later treat the duration as uncertain (a distribution
# around this most-likely value) rather than fixed.

# %%
# Task definitions: id -> (name, most_likely_duration, resource_requirement, predecessors)
tasks = {
    0: ("Mobilization", 2, 1, []),
    1: ("Site clearing", 3, 2, [0]),
    2: ("Excavation", 4, 3, [1]),
    3: ("Foundation formwork", 3, 2, [2]),
    4: ("Foundation pour", 2, 2, [3]),
    5: ("Foundation cure", 4, 0, [4]),           # curing needs zero crew
    6: ("Backfill", 2, 2, [5]),
    7: ("Framing - ground flr", 5, 3, [5]),
    8: ("Framing - upper flr", 5, 3, [7]),
    9: ("Roofing", 3, 2, [8]),
    10: ("Rough plumbing", 3, 2, [7]),
    11: ("Rough electrical", 3, 2, [7]),
    12: ("Insulation", 2, 1, [10, 11, 9]),
    13: ("Drywall", 4, 2, [12]),
    14: ("Interior finishes", 5, 2, [13]),
    15: ("Exterior finishes", 4, 2, [9, 6]),
    16: ("Final inspection", 1, 1, [14, 15]),
}

n_tasks = len(tasks)
base_durations = {i: v[1] for i, v in tasks.items()}
resource_req = {i: v[2] for i, v in tasks.items()}
predecessors = {i: v[3] for i, v in tasks.items()}
names = {i: v[0] for i, v in tasks.items()}

RESOURCE_CAPACITY = 4  # max crew units that can work in parallel across all tasks

print(f"Project has {n_tasks} tasks, resource capacity = {RESOURCE_CAPACITY} crew units")
task_table = pd.DataFrame({
    "Task": names,
    "Most likely duration": base_durations,
    "Resource req.": resource_req,
    "Predecessors": predecessors
})
print(task_table)

# %% [markdown]
# ## 3. Visualize the precedence network

# %%
G = nx.DiGraph()
for i in tasks:
    G.add_node(i, label=names[i])
for i, preds in predecessors.items():
    for p in preds:
        G.add_edge(p, i)

pos = nx.spring_layout(G, seed=7, k=1.2)
plt.figure(figsize=(12, 7))
nx.draw(G, pos, with_labels=False, node_color="#9ecae1", node_size=1400,
        arrowsize=18, edge_color="#666666")
labels = {i: f"{i}\n{names[i]}" for i in tasks}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
plt.title("Construction project precedence network")
plt.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Baseline — Critical Path Method (CPM)
#
# CPM computes the minimum possible project duration **assuming unlimited
# resources and fixed durations** — every eligible task starts the moment its
# predecessors finish. This is a theoretical lower bound that no
# resource-constrained, uncertain-duration schedule can beat.

# %%
def compute_cpm(durations, predecessors):
    """Forward pass: earliest start/finish times ignoring resource limits."""
    order = list(nx.topological_sort(G))
    earliest_start = {}
    earliest_finish = {}
    for t in order:
        if not predecessors[t]:
            earliest_start[t] = 0
        else:
            earliest_start[t] = max(earliest_finish[p] for p in predecessors[t])
        earliest_finish[t] = earliest_start[t] + durations[t]
    project_duration = max(earliest_finish.values())
    return earliest_start, earliest_finish, project_duration


cpm_start, cpm_finish, cpm_duration = compute_cpm(base_durations, predecessors)
print(f"CPM (unconstrained, deterministic) minimum project duration: {cpm_duration} days")

cpm_df = pd.DataFrame({
    "Task": names,
    "Start": cpm_start,
    "Finish": cpm_finish
}).sort_values("Start")
print(cpm_df)

# %% [markdown]
# ## 5. The scheduling environment (resource-constrained simulator)
#
# This simulator enforces precedence and resource-capacity rules exactly like a
# real site would: a task can only start once its predecessors are finished AND
# enough crew capacity is free. Every priority rule below uses this same
# simulator, so comparisons are apples-to-apples — only the *decision rule* for
# picking which eligible task to start next differs between methods.

# %%
class RCPSPEnv:
    def __init__(self, durations, resource_req, predecessors, capacity):
        self.durations = durations
        self.resource_req = resource_req
        self.predecessors = predecessors
        self.capacity = capacity
        self.n = len(durations)
        self.reset()

    def reset(self):
        self.t = 0
        self.completed = set()
        self.in_progress = {}  # task_id -> remaining_duration
        self.used_resources = 0
        self.start_times = {}
        self.finish_times = {}

    def eligible_tasks(self):
        """Tasks whose predecessors are all completed, not yet started, are eligible."""
        started_or_done = set(self.in_progress.keys()) | self.completed
        elig = []
        for i in range(self.n):
            if i in started_or_done:
                continue
            if all(p in self.completed for p in self.predecessors[i]):
                elig.append(i)
        return elig

    def feasible_tasks(self):
        """Eligible tasks that also fit in remaining resource capacity right now."""
        avail = self.capacity - self.used_resources
        return [i for i in self.eligible_tasks() if self.resource_req[i] <= avail]

    def start_task(self, task_id):
        self.in_progress[task_id] = self.durations[task_id]
        self.used_resources += self.resource_req[task_id]
        self.start_times[task_id] = self.t

    def step_time(self):
        """Advance one day: decrement in-progress tasks, free resources on completion."""
        finished_now = []
        for tid in list(self.in_progress.keys()):
            self.in_progress[tid] -= 1
            if self.in_progress[tid] <= 0:
                finished_now.append(tid)
        for tid in finished_now:
            del self.in_progress[tid]
            self.used_resources -= self.resource_req[tid]
            self.completed.add(tid)
            self.finish_times[tid] = self.t + 1
        self.t += 1

    def is_done(self):
        return len(self.completed) == self.n

# %% [markdown]
# ## 6. Priority-rule heuristics (resource-constrained, deterministic durations)
#
# Three standard, simple priority rules used in practice and in RCPSP literature:
#
# - **Shortest task first** — start whichever feasible task has the shortest duration
# - **Most successors first** — prioritize tasks that unblock the most future work
#   (a common "critical-path-aware" rule)
# - **Most resource-intensive first** — start the biggest resource consumers first
#   while capacity is available

# %%
# Precompute number of successors for each task (used by the "most successors" rule)
successors_count = {i: 0 for i in tasks}
for i, preds in predecessors.items():
    for p in preds:
        successors_count[p] += 1


def run_heuristic(durations, rule):
    env = RCPSPEnv(durations, resource_req, predecessors, RESOURCE_CAPACITY)
    while not env.is_done():
        feasible = env.feasible_tasks()
        while feasible:
            if rule == "shortest_first":
                feasible.sort(key=lambda i: durations[i])
            elif rule == "most_successors_first":
                feasible.sort(key=lambda i: -successors_count[i])
            elif rule == "most_resource_first":
                feasible.sort(key=lambda i: -resource_req[i])
            chosen = feasible[0]
            env.start_task(chosen)
            feasible = env.feasible_tasks()
        env.step_time()
    return env.t, env.start_times.copy(), env.finish_times.copy()


rules = ["shortest_first", "most_successors_first", "most_resource_first"]
deterministic_results = {}
for rule in rules:
    duration, start_times, finish_times = run_heuristic(base_durations, rule)
    deterministic_results[rule] = (duration, start_times, finish_times)
    print(f"{rule:25s} -> makespan = {duration} days (deterministic durations)")

# %% [markdown]
# ## 7. Introducing uncertainty: stochastic task durations
#
# In reality, durations are not fixed numbers — weather, labor availability, and
# material delivery introduce variability. We model each task's duration as a
# **triangular distribution**: minimum, most-likely (our original value), and
# maximum. This is the same distribution commonly used in PERT/risk-based
# construction scheduling, so it's a recognizable, defensible modeling choice.

# %%
# Define (min, most_likely, max) duration for each task - +/- variability
duration_distributions = {}
for i, d in base_durations.items():
    low = max(1, round(d * 0.7))
    high = round(d * 1.5)
    duration_distributions[i] = (low, d, high)

dist_df = pd.DataFrame(duration_distributions, index=["min", "most_likely", "max"]).T
dist_df.index.name = "task_id"
dist_df["name"] = [names[i] for i in dist_df.index]
print(dist_df)


def sample_durations(duration_distributions, rng):
    """Draw one random duration per task from its triangular distribution."""
    sampled = {}
    for i, (low, mode, high) in duration_distributions.items():
        if low == high:
            sampled[i] = low
        else:
            sampled[i] = int(round(rng.triangular(low, mode, high)))
    return sampled

# %% [markdown]
# ## 8. Monte Carlo simulation: makespan distribution per priority rule
#
# For each priority rule, we run the resource-constrained simulator thousands of
# times, each time with a fresh random draw of task durations. This produces a
# full distribution of possible project completion times per rule, not just a
# single number — letting us answer questions like "what's the probability we
# finish within 30 days using this rule?"

# %%
N_SIMULATIONS = 3000
rng = np.random.default_rng(42)

mc_results = {rule: [] for rule in rules}

for sim in range(N_SIMULATIONS):
    sampled_durations = sample_durations(duration_distributions, rng)
    for rule in rules:
        duration, _, _ = run_heuristic(sampled_durations, rule)
        mc_results[rule].append(duration)

mc_df = pd.DataFrame(mc_results)
print(mc_df.describe())

# %% [markdown]
# ## 9. Compare rules: expected makespan, variability, and probability of
# meeting a deadline

# %%
target_deadline = 55  # days - set just above the median outcome, so the comparison is informative

summary_rows = []
for rule in rules:
    values = np.array(mc_results[rule])
    summary_rows.append({
        "Rule": rule,
        "Mean makespan": values.mean(),
        "Std dev": values.std(),
        "P10 (optimistic)": np.percentile(values, 10),
        "P90 (pessimistic)": np.percentile(values, 90),
        f"P(finish <= {target_deadline} days)": (values <= target_deadline).mean(),
    })

summary_df = pd.DataFrame(summary_rows).set_index("Rule")
print(summary_df)

# %%
plt.figure(figsize=(10, 6))
for rule in rules:
    plt.hist(mc_results[rule], bins=25, alpha=0.5, label=rule)
plt.axvline(cpm_duration, color="black", linestyle="--", label=f"CPM lower bound ({cpm_duration} days)")
plt.axvline(target_deadline, color="red", linestyle=":", label=f"Target deadline ({target_deadline} days)")
plt.xlabel("Project makespan (days)")
plt.ylabel("Frequency across simulations")
plt.title(f"Monte Carlo makespan distributions by priority rule (n={N_SIMULATIONS} runs each)")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Sensitivity analysis: which task's duration uncertainty drives overall risk?
#
# Not all 17 tasks contribute equally to the spread in project makespan — tasks on
# or near the critical path matter much more than slack tasks. We quantify this
# with **Spearman rank correlation** between each task's sampled duration and the
# resulting makespan, across all Monte Carlo runs — a simple, standard global
# sensitivity measure that tells us which tasks are actually worth de-risking
# (e.g. adding buffer, assigning more reliable crews, prioritizing procurement)
# versus which barely matter.

# %%
from scipy.stats import spearmanr

# Re-run simulations, but this time keep the per-task sampled durations around
N_SENS = 2000
rng_sens = np.random.default_rng(7)

sampled_matrix = np.zeros((N_SENS, n_tasks))
makespans_sens = np.zeros(N_SENS)

for sim in range(N_SENS):
    sampled = sample_durations(duration_distributions, rng_sens)
    sampled_matrix[sim, :] = [sampled[i] for i in range(n_tasks)]
    makespan, _, _ = run_heuristic(sampled, "shortest_first")
    makespans_sens[sim] = makespan

corrs = []
for i in range(n_tasks):
    rho, _ = spearmanr(sampled_matrix[:, i], makespans_sens)
    corrs.append(rho)

sensitivity_df = pd.DataFrame({
    "Task": [names[i] for i in range(n_tasks)],
    "Spearman correlation with makespan": corrs
}).sort_values("Spearman correlation with makespan", key=abs, ascending=False)

print(sensitivity_df)

# %%
plot_df = sensitivity_df.sort_values("Spearman correlation with makespan")

plt.figure(figsize=(9, 7))
colors = ["#de2d26" if v > 0 else "#3182bd" for v in plot_df["Spearman correlation with makespan"]]
plt.barh(plot_df["Task"], plot_df["Spearman correlation with makespan"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Spearman correlation with project makespan")
plt.title("Which task durations drive overall project risk most")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Schedule visualization (Gantt charts, deterministic case for reference)

# %%
def plot_gantt(start_times, finish_times, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in sorted(start_times.keys()):
        ax.barh(names[i], finish_times[i] - start_times[i], left=start_times[i],
                color="#4292c6", edgecolor="black")
    ax.set_xlabel("Day")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


plot_gantt(cpm_start, cpm_finish, f"CPM schedule (unconstrained, deterministic, {cpm_duration} days)")

best_rule_deterministic = min(rules, key=lambda r: deterministic_results[r][0])
_, best_start, best_finish = deterministic_results[best_rule_deterministic]
plot_gantt(best_start, best_finish,
           f"Best deterministic heuristic: {best_rule_deterministic} "
           f"({deterministic_results[best_rule_deterministic][0]} days)")

# %% [markdown]
# ## 12. Sensitivity check: what if we lose a crew?
#
# A realistic risk scenario: one crew unit becomes unavailable partway through the
# project (e.g. pulled to another site). We re-run the Monte Carlo comparison
# under reduced capacity to see how each rule's risk profile shifts — this is the
# kind of "what-if" analysis a project planner would actually want.

# %%
reduced_capacity = RESOURCE_CAPACITY - 1


def run_heuristic_capacity(durations, rule, capacity):
    env = RCPSPEnv(durations, resource_req, predecessors, capacity)
    while not env.is_done():
        feasible = env.feasible_tasks()
        while feasible:
            if rule == "shortest_first":
                feasible.sort(key=lambda i: durations[i])
            elif rule == "most_successors_first":
                feasible.sort(key=lambda i: -successors_count[i])
            elif rule == "most_resource_first":
                feasible.sort(key=lambda i: -resource_req[i])
            chosen = feasible[0]
            env.start_task(chosen)
            feasible = env.feasible_tasks()
        env.step_time()
    return env.t


mc_results_reduced = {rule: [] for rule in rules}
rng2 = np.random.default_rng(123)
for sim in range(N_SIMULATIONS):
    sampled_durations = sample_durations(duration_distributions, rng2)
    for rule in rules:
        duration = run_heuristic_capacity(sampled_durations, rule, reduced_capacity)
        mc_results_reduced[rule].append(duration)

summary_rows_reduced = []
for rule in rules:
    values = np.array(mc_results_reduced[rule])
    summary_rows_reduced.append({
        "Rule": rule,
        "Mean makespan (reduced capacity)": values.mean(),
        f"P(finish <= {target_deadline} days)": (values <= target_deadline).mean(),
    })

reduced_df = pd.DataFrame(summary_rows_reduced).set_index("Rule")
reduced_df["Mean makespan (nominal capacity)"] = [summary_df.loc[r, "Mean makespan"] for r in reduced_df.index]
reduced_df["Increase from losing 1 crew unit (days)"] = (
    reduced_df["Mean makespan (reduced capacity)"] - reduced_df["Mean makespan (nominal capacity)"]
)
print(reduced_df)

# %%
spread_nominal = summary_df["Mean makespan"].max() - summary_df["Mean makespan"].min()
spread_reduced = reduced_df["Mean makespan (reduced capacity)"].max() - reduced_df["Mean makespan (reduced capacity)"].min()

print(f"Spread between best and worst rule at nominal capacity: {spread_nominal:.2f} days")
print(f"Spread between best and worst rule at reduced capacity: {spread_reduced:.2f} days")
print()
if spread_reduced < spread_nominal:
    print("The rules become MORE similar to each other under tighter resource constraints -- "
          "once capacity is the binding constraint, the order tasks are started in matters less, "
          "because almost every ordering ends up bottlenecked at the same resource ceiling.")
else:
    print("The rules become MORE differentiated under tighter resource constraints -- "
          "ordering choices matter more when there is less slack to absorb delays.")

# %% [markdown]
# ## 13. Discussion / takeaways
#
# - **CPM gives the theoretical floor** (43 days), but it's not achievable with
#   only 4 crew units and uncertain durations — it's a useful reference point,
#   not a usable schedule.
# - **Priority rules perform similarly to each other** under nominal capacity,
#   which is expected — for small RCPSP instances, simple heuristics are already
#   strong. The real value added here is the Monte Carlo layer: instead of
#   reporting one number, we can say "using this rule, there is an X% chance of
#   finishing within the target deadline" — what stakeholders actually need for
#   risk-informed decisions.
# - **Capacity loss makes the rules converge further, not diverge** — under
#   tighter resource constraints, almost every ordering ends up bottlenecked at
#   the same resource ceiling, so the choice of priority rule matters less, not
#   more, when capacity is the binding constraint. This is a more useful (and
#   more honest) finding than assuming a "smarter" rule would pull ahead under
#   stress: it tells a planner that adding crew capacity back is likely to do
#   more for the schedule than re-optimizing task order.
# - **The sensitivity analysis (Section 10) is the most useful output for a
#   planner**: it identifies which specific tasks' uncertainty actually drives
#   overall schedule risk, rather than treating all 17 tasks as equally
#   important — these are the tasks worth prioritizing for buffer, reliable
#   crews, or early procurement.
# - **Honest limitation:** this uses independent triangular sampling per task;
#   in reality, delays are often correlated (e.g. one bad weather week affects
#   several outdoor tasks at once). Correlated sampling would be a natural and
#   credible extension.
#
# ## Possible extensions
# - Correlated duration sampling (e.g. shared weather-delay factor across
#   outdoor tasks)
# - Multiple resource types (labor vs. equipment) instead of one pooled resource
# - Cost-based objective (cost of delay vs. cost of adding a crew) instead of
#   pure makespan minimization
# - Benchmark against a PSPLIB instance to compare against published
#   heuristic/optimal results
