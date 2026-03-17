This project implements a Q-Learning agent that learns how to complete a delivery task in a grid environment.



 ### Environment

	•	Grid: 10x10
	•	Start: (0, 0)
	•	Pickup: (2, 4)
	•	Drop-off: (8, 9)

### Actions

	•	0: Up
	•	1: Down
	•	2: Left
	•	3: Right

### Rewards

	•	Step: -1
	•	Invalid move: -10
	•	Pickup: +20
	•	Delivery: +100


### Agent

	•	Algorithm: Q-Learning
	•	Alpha: 0.1
	•	Gamma: 0.95
	•	Epsilon-greedy exploration



### Convergence

Convergence is detected using:

	•	Variance of recent steps
	•	Change in moving average

This ensures the agent reaches stable performance, not just good performance.

### Learning Curve
<p align="center">
  <img src="learning_curve.png" width="600">
</p>
The graph shows the learnig progress of the agent over episodes.

### Run

Train the agent:
```bash
python main.py
```
Plot results:

```bash
python plot_graph.py
