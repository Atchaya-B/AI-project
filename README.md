# AI-project
# Smart Plant Monitoring System

## Project Overview
This project uses three algorithms to monitor plant conditions from daily soil moisture readings:

- Hidden Markov Model (HMM)
- Kalman Filter
- Dynamic Bayesian Network (DBN)

**Data used in all three programs:** soil moisture (%) over eight days: 65, 58, 42, 35, 49, 60, 53, 40.

---

## Problem Statement 1: HMM
Classify each reading as **Wet** (55% or above), **Medium** (45%-54%) or **Dry** (below 45%), then use the **Viterbi algorithm** to find the most likely hidden plant state (Healthy or Stressed) for each day.

Initial probabilities: P(Healthy) = 0.8, P(Stressed) = 0.2

| Previous State | Healthy | Stressed |
|---|---|---|
| Healthy | 0.85 | 0.15 |
| Stressed | 0.35 | 0.65 |

| State | Wet | Medium | Dry |
|---|---|---|---|
| Healthy | 0.6 | 0.3 | 0.1 |
| Stressed | 0.1 | 0.3 | 0.6 |

**Result**

| Day | Moisture | Category | Most likely state |
|---|---|---|---|
| 1 | 65 | Wet | Healthy |
| 2 | 58 | Wet | Healthy |
| 3 | 42 | Dry | Stressed |
| 4 | 35 | Dry | Stressed |
| 5 | 49 | Medium | Healthy |
| 6 | 60 | Wet | Healthy |
| 7 | 53 | Medium | Healthy |
| 8 | 40 | Dry | Stressed |

Sequence: Healthy -> Healthy -> Stressed -> Stressed -> Healthy -> Healthy -> Healthy -> Stressed

---

## Problem Statement 2: Kalman Filter
Estimate the soil moisture each day with a Kalman filter.

- Initial estimate = 60
- Initial uncertainty P = 10
- Process noise variance Q = 2
- Measurement noise variance R = 9

Each day: **predict** (P = P + Q), then **update** (K = P / (P + R), estimate = estimate + K x (measurement - estimate), P = (1 - K) x P).

**Result**

| Day | Measured | Estimate |
|---|---|---|
| 1 | 65 | 62.86 |
| 2 | 58 | 60.71 |
| 3 | 42 | 53.24 |
| 4 | 35 | 46.25 |
| 5 | 49 | 47.29 |
| 6 | 60 | 52.05 |
| 7 | 53 | 52.41 |
| 8 | 40 | 47.77 |

**Final estimate: 47.77%**

---

## Problem Statement 3: DBN
Calculate the probability of plant stress each day with a simplified Dynamic Bayesian Network.

Watering status: False, False, False, True, False, False, False, True. Initial probability of stress = 0.10.

| Moisture | Healthy to Stressed | Stressed to Stressed |
|---|---|---|
| Wet | 0.05 | 0.25 |
| Medium | 0.15 | 0.55 |
| Dry | 0.60 | 0.85 |

When watering occurs, subtract 0.20 from the Healthy-to-Stressed probability and 0.25 from the Stressed-to-Stressed probability, with a minimum of 0.01.

Each day: `P(Stressed) = P(previous) x P(S to S) + (1 - P(previous)) x P(H to S)`. The plant is **Stressed** if the probability is at least 0.50, otherwise **Healthy**.

**Result**

| Day | Moisture | Watered | P(Stressed) | Class |
|---|---|---|---|---|
| 1 | 65 | No | 0.0700 | Healthy |
| 2 | 58 | No | 0.0640 | Healthy |
| 3 | 42 | No | 0.6160 | Stressed |
| 4 | 35 | Yes | 0.5232 | Stressed |
| 5 | 49 | No | 0.3593 | Healthy |
| 6 | 60 | No | 0.1219 | Healthy |
| 7 | 53 | No | 0.1987 | Healthy |
| 8 | 40 | Yes | 0.4397 | Healthy |

---

## Assumptions
- **Kalman filter:** the moisture is modelled as a constant value with random change (a random walk), and the prediction step is applied before every measurement, including day 1.
- **DBN:** the initial stress probability of 0.10 is the belief before day 1. Each day's moisture category and watering status are used to choose the transition probabilities for that same day.
- **Moisture boundaries:** 55 counts as Wet, 45 counts as Medium.

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
python plant_hmm.py
python plant_kalman.py
python plant_dbn.py
```

Each program prints a table and saves a graph (`hmm_result.png`, `kalman_result.png`, `dbn_result.png`).

## Note
This is an educational project using sample data and illustrative probabilities. The predictions have not been validated against real plant measurements.
