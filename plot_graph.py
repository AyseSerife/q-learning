import matplotlib.pyplot as plt
import csv


def plot_learning_curve(data_file="graph_data.txt"):
    episodes = []
    steps = []
    variances = []

    # Read data file
    try:
        with open(data_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header

            for row in reader:
                if len(row) >= 2:
                    episodes.append(int(row[0]))
                    steps.append(int(row[1]))

                    # If variance is not empty
                    if len(row) >= 3 and row[2] != "":
                        variances.append(float(row[2]))
                    else:
                        variances.append(None)  # Pending

    except FileNotFoundError:
        print(f"Error: '{data_file}' not found.")
        return

    # Plot the graph
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary Axis: Raw Steps
    ax1.plot(episodes, steps, label='Raw Steps', color='darkgray', alpha=0.7)
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Steps", fontsize=12, color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    # Secondary Axis: Variance
    ax2 = ax1.twinx()

    valid_episodes = [e for e, v in zip(episodes, variances) if v is not None]
    valid_variances = [v for v in variances if v is not None]

    ax2.plot(valid_episodes, valid_variances, label='Variance', color='blue', linewidth=2, alpha=0.5)
    ax2.set_ylabel("Variance", fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue',)

    plt.title("Q-Learning Stochastic Learning Curve", fontsize=16)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    ax1.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("learning_curve.png")
    plt.show()


if __name__ == "__main__":
    plot_learning_curve()