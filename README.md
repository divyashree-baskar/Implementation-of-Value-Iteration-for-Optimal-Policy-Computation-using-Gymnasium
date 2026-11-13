# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement

Implement the Value Iteration algorithm to solve the Markov Decision Process (MDP) for a customized FrozenLake environment. The algorithm should compute the optimal state-value function and derive the optimal policy that maximizes the expected cumulative reward.




## Software Requirements
```text
1.Python 3.x
2.Gymnasium
3.NumPy
4.Matplotlib
5.Jupyter Notebook / VS Code
```
Installation:
```text
pip install gymnasium numpy matplotlib
```
## Environment Description

We use a custom 4×4 FrozenLake environment.
```text
F F F G
F H F F
F F H F
S F F F
```
Where:

S → Start State
F → Frozen State (Safe)
H → Hole
G → Goal State

Actions:

0 → Left (L)
1 → Down (D)
2 → Right (R)
3 → Up (U)

Rewards:

Goal = +1
Hole = −1
Frozen = 0

## MDP Representation

A Markov Decision Process consists of:

States (S): 16 grid positions.
Actions (A): Left, Down, Right, Up.
Transition Probability (P): Probability of reaching the next state.
Reward (R): Immediate reward received.
Discount Factor (γ): Determines the importance of future rewards.

## Theory
alue Iteration is a Dynamic Programming algorithm used to determine the optimal value of every state.

Instead of evaluating a fixed policy, Value Iteration repeatedly updates the value function using the Bellman Optimality Equation.


## Algorithm

1.Create the customized FrozenLake environment.
2.Modify rewards for holes and goal.
3.Initialize all state values to zero.
4.Repeat:
      Compute the value of every action.
      Select the maximum action value.
      Update the state value.
5.Stop when values converge.
6.Extract the optimal policy.
7.Display the optimal value function and policy.



## Python Program

```python

# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------
# Write your code here
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env_desc = [
    "FFFG",
    "FHFF",
    "FFHF",
    "SFFF"
]

env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)

rows = len(env_desc)
cols = len(env_desc[0])

# -------------------------------------------------
# Modify Rewards
# -------------------------------------------------

for state in range(env.observation_space.n):
    for action in range(env.action_space.n):

        transitions = env.unwrapped.P[state][action]
        new_transitions = []

        for prob, next_state, reward, done in transitions:

            row = next_state // cols
            col = next_state % cols

            if env_desc[row][col] == "H":
                reward = -1

            elif env_desc[row][col] == "G":
                reward = 1

            else:
                reward = 0

            new_transitions.append((prob, next_state, reward, done))

        env.unwrapped.P[state][action] = new_transitions
def value_iteration(env, gamma=0.99, theta=1e-8):

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    V = np.zeros(num_states)

    iteration = 0

    while True:

        delta = 0

        for s in range(num_states):

            action_values = []

            for a in range(num_actions):

                value = 0

                for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                    value += prob * (reward + gamma * V[next_state])

                action_values.append(value)

            best_value = max(action_values)

            delta = max(delta, abs(best_value - V[s]))

            V[s] = best_value

        iteration += 1

        if delta < theta:
            break

    # -----------------------------
    # Extract Optimal Policy
    # -----------------------------

    policy = np.zeros(num_states, dtype=int)

    for s in range(num_states):

        action_values = []

        for a in range(num_actions):

            value = 0

            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                value += prob * (reward + gamma * V[next_state])

            action_values.append(value)

        policy[s] = np.argmax(action_values)

    return V, policy, iteration

V, policy, iterations = value_iteration(env)
print("Name : DIVYASHREE B")
print("Register Number : 212224040081 ")

print("\nValue Iteration Completed")

print("\nNumber of Iterations:", iterations)

print("\nOptimal State-Value Function:\n")
print(np.round(V.reshape(rows, cols), 4))

action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[a] for a in policy]
).reshape(rows, cols)

print("\nOptimal Policy:\n")
print(policy_grid)

env.close()



```

---

## Output

```text

Name : DIVYASHREE B
Register Number : 212224040081 

Value Iteration Completed

Number of Iterations: 1834

Optimal State-Value Function:

[[  85.1373   87.7172   92.9553  100.    ]
 [  79.3027 -100.       28.0171   92.9553]
 [  75.8711   16.7416 -100.       87.7172]
 [  74.7387   75.8711   79.3027   85.1373]]

Optimal Policy:

[['U' 'U' 'U' 'L']
 ['L' 'L' 'R' 'R']
 ['L' 'D' 'L' 'R']
 ['R' 'D' 'D' 'R']]

```
<img width="526" height="442" alt="Screenshot 2026-11-13 212033" src="https://github.com/user-attachments/assets/25f9333b-ec5e-4d07-b7c9-a7d8ade57bde" />



---

## Result
```text

The Value Iteration algorithm was successfully implemented on the customized FrozenLake environment. The algorithm computed the optimal state-value function and derived the optimal policy using the Bellman Optimality Equation.
```
---

## Inference
```text
The Value Iteration algorithm successfully computed the optimal state-value function and policy for both the original and modified FrozenLake environments. The original environment converged in 324 iterations, whereas the modified environment took 1834 iterations due to the changed start state, goal state, hole positions, and negative hole reward. This comparison shows that changes in the environment and reward structure significantly affect the optimal policy, state values, and convergence time, demonstrating the adaptability of the Value Iteration algorithm.

```
---

