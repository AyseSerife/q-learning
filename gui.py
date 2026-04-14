import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import time
import csv
import numpy as np

# Matplotlib Tkinter entegrasyonu için gerekli kütüphaneler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from environment import DeliveryEnvironment
from agent import QLearningAgent


class QLearningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stochastic Q-Learning Simulation")

        self.env = DeliveryEnvironment()
        self.agent = None

        self.grid_size = self.env.grid_size
        self.cell_size = 50

        self.action_names = {
            0: "Up", 1: "Down", 2: "Left", 3: "Right",
            4: "Up-Right", 5: "Down-Right", 6: "Up-Left", 7: "Down-Left", 8: "Stay"
        }

        self.create_widgets()
        self.draw_initial_grid()

    def create_widgets(self):
        # Left Panel - Controls
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(control_frame, text="Episodes:").pack(pady=5)
        self.episodes_entry = tk.Entry(control_frame)
        self.episodes_entry.insert(0, "10000")
        self.episodes_entry.pack(pady=5)

        tk.Label(control_frame, text="Max Epsilon:").pack(pady=5)
        self.max_eps_entry = tk.Entry(control_frame)
        self.max_eps_entry.insert(0, "1.0")
        self.max_eps_entry.pack(pady=5)

        tk.Label(control_frame, text="Min Epsilon:").pack(pady=5)
        self.min_eps_entry = tk.Entry(control_frame)
        self.min_eps_entry.insert(0, "0.01")
        self.min_eps_entry.pack(pady=5)

        tk.Label(control_frame, text="Min Prob (Center):").pack(pady=5)
        self.min_prob_entry = tk.Entry(control_frame)
        self.min_prob_entry.insert(0, "0.70")
        self.min_prob_entry.pack(pady=5)

        tk.Label(control_frame, text="Max Prob (Edges):").pack(pady=5)
        self.max_prob_entry = tk.Entry(control_frame)
        self.max_prob_entry.insert(0, "0.95")
        self.max_prob_entry.pack(pady=5)

        self.train_btn = tk.Button(control_frame, text="Fast Train", command=self.start_training)
        self.train_btn.pack(pady=15)

        self.test_btn = tk.Button(control_frame, text="Show Optimal Path", command=self.start_testing,
                                  state=tk.DISABLED)
        self.test_btn.pack(pady=10)

        # Yeni Grafik Gösterme Butonu
        self.plot_btn = tk.Button(control_frame, text="Show Learning Curve", command=self.show_plot, state=tk.DISABLED)
        self.plot_btn.pack(pady=10)

        # Right Panel - Grid & Logs
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(right_frame, width=self.grid_size * self.cell_size,
                                height=self.grid_size * self.cell_size, bg="white")
        self.canvas.pack(pady=10)

        self.log_text = scrolledtext.ScrolledText(right_frame, height=12, width=80)
        self.log_text.pack(pady=10)

    def get_heat_color(self, x, y):
        # Color gradient by Euclidean distance
        dist = np.sqrt((self.env.zone_center[0] - x) ** 2 + (self.env.zone_center[1] - y) ** 2)
        max_dist = np.sqrt(8 ** 2 + 7 ** 2)
        intensity = max(0, 1 - (dist / max_dist))

        r = 255
        g = int(255 * (1 - intensity * 0.7))
        b = int(255 * (1 - intensity * 0.7))
        return f'#{r:02x}{g:02x}{b:02x}'

    def draw_initial_grid(self):
        self.canvas.delete("all")
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                x1 = y * self.cell_size
                y1 = x * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                color = self.get_heat_color(x, y)
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

        px, py = self.env.pickup_pos
        self.canvas.create_rectangle(py * self.cell_size, px * self.cell_size, (py + 1) * self.cell_size,
                                     (px + 1) * self.cell_size, fill="green")
        self.canvas.create_text(py * self.cell_size + 25, px * self.cell_size + 25, text="PICKUP", fill="white",
                                font=("Arial", 9, "bold"), justify=tk.CENTER)

        dx, dy = self.env.dropoff_pos
        self.canvas.create_rectangle(dy * self.cell_size, dx * self.cell_size, (dy + 1) * self.cell_size,
                                     (dx + 1) * self.cell_size, fill="blue")
        self.canvas.create_text(dy * self.cell_size + 25, dx * self.cell_size + 25, text="DROPOFF", fill="white",
                                font=("Arial", 9, "bold"), justify=tk.CENTER)

    def update_truck(self, state):
        self.canvas.delete("truck")
        x, y, has_load = state
        y1, x1 = x * self.cell_size, y * self.cell_size

        color = "orange" if has_load else "yellow"
        self.canvas.create_oval(x1 + 10, y1 + 10, x1 + 40, y1 + 40, fill=color, tags="truck")
        self.canvas.create_text(x1 + 25, y1 + 25, text="T", font=("Arial", 12, "bold"), tags="truck")

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def start_training(self):
        try:
            episodes = int(self.episodes_entry.get())
            eps_start = float(self.max_eps_entry.get())
            eps_end = float(self.min_eps_entry.get())
            min_prob = float(self.min_prob_entry.get())
            max_prob = float(self.max_prob_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers.")
            return

        # Ajanı eğitmeden hemen önce evrenin olasılık matematiğini güncelle!
        self.env.update_probabilities(min_prob, max_prob)

        self.agent = QLearningAgent(self.env, episodes, eps_start, eps_end)

        self.train_btn.config(state=tk.DISABLED)
        self.plot_btn.config(state=tk.DISABLED)
        self.log(f"Training started: {episodes} Episodes, Epsilon {eps_start} -> {eps_end}")
        self.log(f"Environment rules updated: Min Prob {min_prob}, Max Prob {max_prob}")

        threading.Thread(target=self.run_training, daemon=True).start()

    def run_training(self):
        self.agent.train()
        self.root.after(0, self.training_finished)

    def training_finished(self):
        self.log("Training finished!")
        self.train_btn.config(state=tk.NORMAL)
        self.test_btn.config(state=tk.NORMAL)
        self.plot_btn.config(state=tk.NORMAL)  # Grafik butonunu aktif et

    def start_testing(self):
        self.test_btn.config(state=tk.DISABLED)
        self.log("\n--- SHOWING OPTIMAL PATH ---")
        threading.Thread(target=self.run_testing, daemon=True).start()

    def run_testing(self):
        state = self.env.reset()
        self.root.after(0, self.draw_initial_grid)
        self.root.after(0, self.update_truck, state)

        done = False
        step = 0

        while not done and step < 50:
            time.sleep(0.3)

            state_idx = self.agent.get_state_index(state)
            intended_action = np.argmax(self.agent.q_table[state_idx])

            next_state, reward, done, actual_action = self.env.step(intended_action)

            msg = f"Step {step + 1}: Intended: {self.action_names[intended_action]:<10} -> Actual: {self.action_names[actual_action]}"
            self.root.after(0, self.log, msg)
            self.root.after(0, self.update_truck, next_state)

            state = next_state
            step += 1

        if done:
            self.root.after(0, self.log, "GOAL REACHED!")
        else:
            self.root.after(0, self.log, "FAILED TO REACH GOAL!")

        self.root.after(0, lambda: self.test_btn.config(state=tk.NORMAL))

    def show_plot(self):
        # Ayrı bir pencere (Toplevel) oluştur
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Learning Curve")
        plot_window.geometry("900x600")

        episodes = []
        steps = []
        variances = []
        moving_averages = []

        try:
            with open("graph_data.txt", 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header

                for row in reader:
                    if len(row) >= 2:
                        episodes.append(int(row[0]))
                        steps.append(int(row[1]))

                        if len(row) >= 4 and row[2] != "" and row[3] != "":
                            variances.append(float(row[2]))
                            moving_averages.append(float(row[3]))
                        else:
                            variances.append(None)
                            moving_averages.append(None)
        except FileNotFoundError:
            messagebox.showerror("Error", "Graph data not found. Please train the agent first.")
            plot_window.destroy()
            return

        # Grafiği çiz
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(episodes, steps, label='Raw Steps', color='lightgray', alpha=0.3)

        valid_episodes = [e for e, m in zip(episodes, moving_averages) if m is not None]
        valid_moving_averages = [m for m in moving_averages if m is not None]

        ax1.plot(valid_episodes, valid_moving_averages, label='Moving Average', color='blue', linewidth=2)

        ax1.set_xlabel("Episode", fontsize=12)
        ax1.set_ylabel("Steps", fontsize=12, color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        ax2 = ax1.twinx()
        valid_variances = [v for v in variances if v is not None]

        ax2.plot(valid_episodes, valid_variances, label='Variance', color='red', linewidth=2, alpha=0.5)
        ax2.set_ylabel("Variance", fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title("Stochastic Learning Curve", fontsize=16)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

        ax1.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()

        # Grafiği Tkinter penceresine göm (Embed plot into Tkinter)
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = QLearningGUI(root)
    root.mainloop()