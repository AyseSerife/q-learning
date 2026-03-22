from statistics import variance

import numpy as np
import random


class QLearningAgent:
    def __init__(self, env, episodes, epsilon_start, epsilon_end):
        self.env = env
        self.episodes = episodes
        self.epsilon = epsilon_start
        self.min_epsilon = epsilon_end

        # Epsilon decay rate
        self.epsilon_decay = (self.min_epsilon / self.epsilon) ** (1 / self.episodes) if self.episodes > 0 else 0.99

        self.alpha = 0.1
        self.gamma = 0.95
        self.q_table = np.zeros((200, 4)) # 10 * 10 = 100 (grid size) 100 * 2 = 200 load-unload

    def get_state_index(self, state):
        x, y, has_load = state
        return (x * 10 * 2) + (y * 2) + has_load

    def train(self, table_file="training_table.txt", graph_file="graph_data.txt"):
        window_size = 10
        recent_steps = []

        consecutive_value = 0
        avg_threshold = 0.5
        var_threshold = 1

        # open both files in write mode
        with open(table_file, "w", encoding="utf-8") as t_file, open(graph_file, "w", encoding="utf-8") as g_file:

            # Table Header
            header = f"| {'Episode':^17} | {'Steps':^15} | {'Variance':^27} |"
            separator = "-" * len(header)
            t_file.write(separator + "\n")
            t_file.write(header + "\n")
            t_file.write(separator + "\n")

            # CSV Format Header
            g_file.write("Episode,Steps,Variance\n")

            for episode in range(self.episodes):
                state = self.env.reset()
                state_idx = self.get_state_index(state)
                done = False
                step_count = 0
                previous_avg = 0
                while not done:
                    # Exploration vs Exploitation
                    if random.uniform(0, 1) < self.epsilon:
                        action = random.choice(self.env.action_space)
                    else:
                        action = np.argmax(self.q_table[state_idx])

                    next_state, reward, done = self.env.step(action)
                    next_state_idx = self.get_state_index(next_state)

                    old_value = self.q_table[state_idx, action]
                    next_max = np.max(self.q_table[next_state_idx])

                    new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
                    self.q_table[state_idx, action] = new_value

                    state_idx = next_state_idx
                    step_count += 1
                    if step_count > 1000:
                        break

                if self.epsilon > self.min_epsilon:
                    self.epsilon *= self.epsilon_decay

                if len(recent_steps) > 0:
                    previous_avg = np.mean(recent_steps)

                recent_steps.append(step_count)

                if len(recent_steps) > window_size:
                    recent_steps.pop(0)

                # Sliding window
                if len(recent_steps) == window_size:
                    moving_average = np.mean(recent_steps)
                    step_variance = variance(recent_steps)


                    # String formatting
                    t_file.write(f"| {episode + 1:^17} | {step_count:^15} | {step_variance:^27.2f} |\n")

                    # Comma-separated format
                    g_file.write(f"{episode + 1},{step_count},{step_variance:.2f}\n")

                    # Dynamic Convergence

                    if step_variance <= var_threshold and (abs(previous_avg - moving_average) <= avg_threshold):
                        consecutive_value += 1
                    else:
                        consecutive_value = 0

                    if consecutive_value >= 30:
                        t_file.write(separator + "\n")
                        print(f"\n--- CONVERGENCE ---")
                        print(f"Agent reached optimum performance at level {episode + 1}.")
                        break
                else:
                    t_file.write(f"| {episode + 1:^17} | {step_count:^15} | {'...':^27} |\n")
                    # For sections where the average has not yet been calculated, we leave that section blank (Missing data)
                    g_file.write(f"{episode + 1},{step_count},\n")

            if len(recent_steps) == window_size and consecutive_value < 30:
                t_file.write(separator + "\n")

    def test_and_save_solution(self, filename="steps.txt"):
        state = self.env.reset()
        done = False
        total_test_steps = 0
        action_names = ["Up", "Down", "Left", "Right"]

        with open(filename, "w", encoding="utf-8") as file:
            file.write("=== Solution Path ===\n\n")

            while not done:
                state_idx = self.get_state_index(state)
                action = np.argmax(self.q_table[state_idx])

                file.write(
                    f"Step: {total_test_steps + 1}: Truck {state[:2]} Action: {action_names[action]}\n")

                state, reward, done = self.env.step(action)
                total_test_steps += 1

                if total_test_steps > 50:
                    file.write("\nERROR: The truck couldn’t find the route and got stuck in a loop.\n")
                    break

            if done:
                file.write(f"\nSUCCESS: Total number of steps: {total_test_steps}\n")