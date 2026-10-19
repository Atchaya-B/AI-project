# Smart River Monitoring System (Rainfall and Water Level)

## Project Overview
This project uses three algorithms to monitor river conditions from daily rainfall and water level readings:

- Hidden Markov Model (HMM)
- Kalman Filter
- Dynamic Bayesian Network (DBN)

**Data used in the programs (8 days):**

| Day | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Rainfall (mm) | 0 | 5 | 35 | 70 | 25 | 3 | 60 | 15 |
| Water level (m) | 2.1 | 2.3 | 3.4 | 4.8 | 4.2 | 3.6 | 4.9 | 4.0 |

---

## Problem Statement 1: HMM
Classify each water level reading as **High** (4.5 m or above), **Medium** (3.0 m to 4.4 m) or **Low** (below 3.0 m), then use the **Viterbi algorithm** to find the most likely hidden river state (Normal or Flood-Risk) for each day.

Initial probabilities: P(Normal) = 0.8, P(Flood-Risk) = 0.2

| Previous State | Normal | Flood-Risk |
|---|---|---|
| Normal | 0.85 | 0.15 |
| Flood-Risk | 0.35 | 0.65 |

| State | Low | Medium | High |
|---|---|---|---|
| Normal | 0.6 | 0.3 | 0.1 |
| Flood-Risk | 0.1 | 0.3 | 0.6 |

**Result**

| Day | Rain (mm) | Level (m) | Category | Most likely state |
|---|---|---|---|---|
| 1 | 0 | 2.1 | Low | Normal |
| 2 | 5 | 2.3 | Low | Normal |
| 3 | 35 | 3.4 | Medium | Normal |
| 4 | 70 | 4.8 | High | Flood-Risk |
| 5 | 25 | 4.2 | Medium | Flood-Risk |
| 6 | 3 | 3.6 | Medium | Flood-Risk |
| 7 | 60 | 4.9 | High | Flood-Risk |
| 8 | 15 | 4.0 | Medium | Flood-Risk |

Sequence: Normal -> Normal -> Normal -> Flood-Risk -> Flood-Risk -> Flood-Risk -> Flood-Risk -> Flood-Risk

---

## Problem Statement 2: Kalman Filter
Estimate the true water level each day with a Kalman filter.

- Initial estimate = 2.0 m
- Initial uncertainty P = 1.0
- Process noise variance Q = 0.1
- Measurement noise variance R = 0.25

Each day: **predict** (P = P + Q), then **update** (K = P / (P + R), estimate = estimate + K x (measurement - estimate), P = (1 - K) x P).

**Result**

| Day | Measured (m) | Estimate (m) |
|---|---|---|
| 1 | 2.1 | 2.08 |
| 2 | 2.3 | 2.20 |
| 3 | 3.4 | 2.79 |
| 4 | 4.8 | 3.73 |
| 5 | 4.2 | 3.95 |
| 6 | 3.6 | 3.79 |
| 7 | 4.9 | 4.30 |
| 8 | 4.0 | 4.16 |

**Final estimate: 4.16 m**

---

## Problem Statement 3: DBN
Calculate the probability of flood risk each day with a simplified Dynamic Bayesian Network, using the rainfall.

Drainage status (gates opened or pumping): False, False, False, True, False, False, False, True. Initial probability of flood risk = 0.10.

Rainfall categories: **Heavy** (30 mm or more), **Moderate** (10 mm to 29 mm), **Light** (below 10 mm).

| Rainfall | Normal to Flood-Risk | Flood-Risk to Flood-Risk |
|---|---|---|
| Light | 0.05 | 0.30 |
| Moderate | 0.20 | 0.55 |
| Heavy | 0.60 | 0.85 |

When drainage occurs, subtract 0.20 from the Normal-to-Flood-Risk probability and 0.25 from the Flood-Risk-to-Flood-Risk probability, with a minimum of 0.01.

Each day: `P(Risk) = P(previous) x P(R to R) + (1 - P(previous)) x P(N to R)`. The river is **Flood-Risk** if the probability is at least 0.50, otherwise **Normal**.

**Result**

| Day | Rain (mm) | Drainage | P(Flood-Risk) | Class |
|---|---|---|---|---|
| 1 | 0 | No | 0.0750 | Normal |
| 2 | 5 | No | 0.0688 | Normal |
| 3 | 35 | No | 0.6172 | Flood-Risk |
| 4 | 70 | Yes | 0.5234 | Flood-Risk |
| 5 | 25 | No | 0.3832 | Normal |
| 6 | 3 | No | 0.1458 | Normal |
| 7 | 60 | No | 0.6365 | Flood-Risk |
| 8 | 15 | Yes | 0.1946 | Normal |

---

## Assumptions
- **HMM:** uses only the water level category as the observation. Rainfall is used in the DBN.
- **Kalman filter:** the level is modelled as a constant value with random change, and the prediction step is applied before every measurement, including day 1.
- **DBN:** the initial probability of 0.10 is the belief before day 1. Each day's rainfall category and drainage status choose the transition probabilities for that same day.
- **Boundaries:** 4.5 m counts as High, 3.0 m counts as Medium, 30 mm counts as Heavy, 10 mm counts as Moderate.

## Technologies Used
- Python
- Matplotlib for graphs

## How to Run
Install Matplotlib:

```
python -m pip install matplotlib
```

Run each program:

```
python river_hmm.py
python river_kalman.py
python river_dbn.py
```

Each program prints a table and saves a graph (`hmm_result.png`, `kalman_result.png`, `dbn_result.png`).

## Note
This is an educational project using sample data and illustrative probabilities. The predictions have not been validated against real river measurements.
