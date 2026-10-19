
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # remove this line if you want pop-up windows
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import logging
logging.getLogger('hmmlearn').setLevel(logging.ERROR)   # hide harmless warnings

np.random.seed(42)

# ----------------------------------------------------------------------
# STEP 1: LOAD OR SIMULATE THE DATA
# ----------------------------------------------------------------------
def simulate_data(n_days=1500):
    """Monsoon-style rain + a river level that rises after rain."""
    days = np.arange(n_days)
    season = np.sin(2 * np.pi * (days % 365) / 365 - 1.2)      # peak in monsoon
    rain_prob = np.clip(0.25 + 0.45 * season, 0.03, 0.9)
    is_rain = np.random.rand(n_days) < rain_prob
    rainfall = is_rain * np.random.exponential(12, n_days)      # mm per day

    true_level = np.zeros(n_days)
    true_level[0] = 2.0
    for t in range(1, n_days):
        # level slowly drains back to 2 m and jumps up with rain
        true_level[t] = (2.0 + 0.92 * (true_level[t - 1] - 2.0)
                         + 0.035 * rainfall[t] + np.random.normal(0, 0.03))
    observed = true_level + np.random.normal(0, 0.15, n_days)   # noisy gauge
    return pd.DataFrame({"rainfall": rainfall, "water_level": observed,
                         "true_level": true_level})


if os.path.exists("river_data.csv"):
    df = pd.read_csv("river_data.csv")[["rainfall", "water_level"]]
    df = df.interpolate().dropna().reset_index(drop=True)
    print("Loaded river_data.csv:", df.shape)
else:
    df = simulate_data()
    print("river_data.csv not found -> using simulated data:", df.shape)

print(df.head(), "\n")
rain = df["rainfall"].values
level = df["water_level"].values
n = len(df)

# ----------------------------------------------------------------------
# STEP 2: KALMAN FILTER  (denoise the level + forecast tomorrow)
# ----------------------------------------------------------------------
# Hidden state = the TRUE water level. We only see a noisy gauge reading.
# Model (rain is the "control input" that pushes the level up):
#     level_t = c + a * level_{t-1} + b * rain_t + process noise
#     reading_t = level_t + sensor noise
# Each day the filter does two things:
#     predict:  x = c + a*x + b*rain          P = a*P*a + Q
#     update :  K = P / (P + R)               x = x + K*(reading - x)
#               (K near 1 = trust the sensor, K near 0 = trust the model)
def fit_dynamics(z, rain):
    """Learn c, a, b by simple least squares: level_t ~ level_{t-1}, rain_t."""
    A = np.column_stack([np.ones(len(z) - 1), z[:-1], rain[1:]])
    c, a, b = np.linalg.lstsq(A, z[1:], rcond=None)[0]
    return c, a, b


def kalman_filter(z, rain, c, a, b, q, r):
    x, P = z[0], 1.0
    filtered, predicted = [], []
    for k in range(len(z)):
        if k > 0:                                    # predict
            x = c + a * x + b * rain[k]
            P = a * a * P + q
        predicted.append(x)                          # forecast BEFORE seeing z[k]
        K = P / (P + r)                              # update
        x = x + K * (z[k] - x)
        P = (1 - K) * P
        filtered.append(x)
    return np.array(filtered), np.array(predicted)


c, a, b = fit_dynamics(level, rain)
print(f"Learned dynamics: level_t = {c:.2f} + {a:.2f}*level_(t-1) + {b:.3f}*rain_t")

# choose the process-noise q that gives the best next-day forecast
r_noise = np.var(np.diff(level)) / 2
best = None
for q in [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
    f_, p_ = kalman_filter(level, rain, c, a, b, q, r_noise)
    e = np.sqrt(np.mean((level[1:] - p_[1:]) ** 2))
    if best is None or e < best[0]:
        best = (e, q, f_, p_)
_, q_best, filtered, predicted = best

rmse = lambda x, y: float(np.sqrt(np.mean((x - y) ** 2)))
rmse_naive = rmse(level[1:], level[:-1])            # "tomorrow = today"
rmse_kalman = rmse(level[1:], predicted[1:])        # Kalman one-day forecast
print("=== KALMAN FILTER ===")
print(f"Next-day forecast RMSE, naive (tomorrow = today): {rmse_naive:.4f}")
print(f"Next-day forecast RMSE, Kalman filter           : {rmse_kalman:.4f}")
if "true_level" in df:
    print(f"Error of raw sensor vs TRUE level : {rmse(level, df['true_level'].values):.4f}")
    print(f"Error of Kalman    vs TRUE level  : {rmse(filtered, df['true_level'].values):.4f}")
print()

# ----------------------------------------------------------------------
# STEP 3: HIDDEN MARKOV MODEL  (find dry / normal / flood states)
# ----------------------------------------------------------------------
X = np.column_stack([rain, filtered])
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

hmm = GaussianHMM(n_components=3, covariance_type="full",
                  n_iter=100, tol=1e-3, random_state=42)
hmm.fit(X_scaled)
raw_states = hmm.predict(X_scaled)                  # Viterbi: best state path

# name the states by their average water level (low -> high)
order = np.argsort([filtered[raw_states == s].mean() for s in range(3)])
rename = {old: new for new, old in enumerate(order)}
states = np.array([rename[s] for s in raw_states])
state_names = ["Dry / Low", "Normal", "Flood / High"]

print("=== HMM ===")
for s in range(3):
    m = states == s
    print(f"{state_names[s]:13s}: {m.sum():4d} days | "
          f"avg rain {rain[m].mean():5.1f} mm | avg level {filtered[m].mean():.2f}")
T = hmm.transmat_[np.ix_(order, order)]
print("\nTransition matrix (row = today's state, column = tomorrow's state):")
print(pd.DataFrame(T.round(2), index=state_names, columns=state_names))
print(f"\nLog-likelihood of the model: {hmm.score(X_scaled):.1f}\n")

# ----------------------------------------------------------------------
# STEP 4: DYNAMIC BAYESIAN NETWORK  (rain & level over time)
# ----------------------------------------------------------------------
# Variables per day: Rain (R) and Level (L), each Low/Medium/High (0/1/2).
# Network structure (repeats every day):
#     R(t-1) -> R(t)          rain today depends on rain yesterday
#     L(t-1) -> L(t)          level today depends on level yesterday
#     R(t)   -> L(t)          rain today pushes the level up
# We learn each conditional probability table (CPT) by counting.
def to_bins(x, edges=None):
    if edges is None:
        edges = np.quantile(x, [1 / 3, 2 / 3])
    return np.digitize(x, edges), edges


split = int(0.8 * n)                                  # 80% train, 20% test
R_bin, r_edges = to_bins(rain)
# many days have 0 rain, so use fixed sensible rain bins: none / light / heavy
r_edges = np.array([0.5, np.quantile(rain[rain > 0.5], 0.6)]) if (rain > 0.5).any() else r_edges
R_bin = np.digitize(rain, r_edges)
L_bin, l_edges = to_bins(filtered[:split])
L_bin = np.digitize(filtered, l_edges)

K_ = 3
a = 1.0                                               # Laplace smoothing
P_R = np.full((K_, K_), a)                            # P(R_t | R_{t-1})
P_L = np.full((K_, K_, K_), a)                        # P(L_t | L_{t-1}, R_t)
for t in range(1, split):
    P_R[R_bin[t - 1], R_bin[t]] += 1
    P_L[L_bin[t - 1], R_bin[t], L_bin[t]] += 1
P_R /= P_R.sum(axis=1, keepdims=True)
P_L /= P_L.sum(axis=2, keepdims=True)

bin_names = ["Low", "Medium", "High"]
print("=== DYNAMIC BAYESIAN NETWORK ===")
print("P(Rain tomorrow | Rain today)   [rows: today none/light/heavy]")
print(pd.DataFrame(P_R.round(2), index=["none", "light", "heavy"],
                   columns=["none", "light", "heavy"]))


def predict_next_level(l_today, r_today):
    """P(L_{t+1}) = sum over R_{t+1} of P(R_{t+1}|R_t) * P(L_{t+1}|L_t, R_{t+1})"""
    p = np.zeros(K_)
    for r_next in range(K_):
        p += P_R[r_today, r_next] * P_L[l_today, r_next]
    return p


correct_dbn = correct_naive = 0
total = 0
chg_total = chg_dbn = chg_known = correct_known = 0          # days where the level class really changes
for t in range(split, n - 1):
    pred = np.argmax(predict_next_level(L_bin[t], R_bin[t]))
    correct_dbn += pred == L_bin[t + 1]
    correct_naive += L_bin[t] == L_bin[t + 1]
    total += 1
    # same prediction, but tomorrow's rain is KNOWN (e.g. from a weather forecast)
    pred_known = np.argmax(P_L[L_bin[t], R_bin[t + 1]])
    correct_known += pred_known == L_bin[t + 1]
    if L_bin[t + 1] != L_bin[t]:
        chg_total += 1
        chg_dbn += pred == L_bin[t + 1]
        chg_known += pred_known == L_bin[t + 1]
print(f"\nNext-day level class accuracy (test set, {total} days)")
print(f"  Naive (tomorrow = today): {correct_naive / total:.1%}")
print(f"  DBN prediction          : {correct_dbn / total:.1%}")

p_ex = predict_next_level(L_bin[-1], R_bin[-1])
print(f"  DBN + tomorrow's rain forecast: {correct_known / total:.1%}")
print(f"  ...on the {chg_total} hard days where the level class changed: "
      f"{chg_known / max(chg_total, 1):.1%}")
print(f"  DBN (no rain forecast) on those same hard days: {chg_dbn / max(chg_total, 1):.1%}")
print("\nExample - forecast for the day after the last record:")
for name, p in zip(bin_names, p_ex):
    print(f"  P(level = {name:6s}) = {p:.2f}")

# ----------------------------------------------------------------------
# STEP 5: PLOTS
# ----------------------------------------------------------------------
fig, ax = plt.subplots(4, 1, figsize=(13, 13))
show = slice(0, min(500, n))                          # first ~500 days

ax[0].bar(range(n)[show], rain[show], color="steelblue")
ax[0].set_ylabel("Rainfall (mm)")
ax[0].set_title("1. Input data: rainfall")

ax[1].plot(level[show], color="lightgray", label="Noisy sensor")
ax[1].plot(filtered[show], color="red", label="Kalman estimate")
if "true_level" in df:
    ax[1].plot(df["true_level"].values[show], "k--", lw=0.8, label="True level")
ax[1].set_ylabel("Water level")
ax[1].set_title("2. Kalman filter: noise removed")
ax[1].legend(loc="upper right")

colors = ["#2ca02c", "#ffbf00", "#d62728"]
ax[2].plot(filtered[show], color="gray", lw=0.8)
for s in range(3):
    idx = np.where(states[show] == s)[0]
    ax[2].scatter(idx, filtered[show][idx], s=8, color=colors[s], label=state_names[s])
ax[2].set_ylabel("Water level")
ax[2].set_title("3. HMM: hidden states found automatically")
ax[2].legend(loc="upper right")

pred_dbn = [np.argmax(predict_next_level(L_bin[t], R_bin[t])) for t in range(split, n - 1)]
ax[3].plot(range(split + 1, n), L_bin[split + 1:], label="Actual", lw=1.5)
ax[3].plot(range(split + 1, n), pred_dbn, label="DBN prediction", lw=1, alpha=0.8)
ax[3].set_yticks([0, 1, 2]); ax[3].set_yticklabels(bin_names)
ax[3].set_xlabel("Day"); ax[3].set_title("4. DBN: next-day level class (test period)")
ax[3].legend(loc="upper right")

plt.tight_layout()
plt.savefig("results.png", dpi=120)
print("\nSaved plot: results.png")