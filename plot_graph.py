import matplotlib.pyplot as plt
import csv


def plot_learning_curve(data_file="graph_data.txt"):
    episodes = []
    steps = []
    variances = []
    moving_averages = []

    # Read data file
    try:
        with open(data_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header

            for row in reader:
                if len(row) >= 2:
                    episodes.append(int(row[0]))
                    steps.append(int(row[1]))

                    # Add if Variance and Moving Average are not empty
                    if len(row) >= 4 and row[2] != "" and row[3] != "":
                        variances.append(float(row[2]))
                        moving_averages.append(float(row[3]))
                    else:
                        variances.append(None)
                        moving_averages.append(None)  # Pending

    except FileNotFoundError:
        print(f"Error: The file '{data_file}' was not found.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary Axis
    ax1.plot(episodes, steps, label='Raw Steps', color='lightgray', alpha=0.3)

    # Filter valid (non-empty) sections
    valid_episodes = [e for e, m in zip(episodes, moving_averages) if m is not None]
    valid_moving_averages = [m for m in moving_averages if m is not None]

    ax1.plot(valid_episodes, valid_moving_averages, label='Moving Average', color='blue',
             linewidth=2)

    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Steps", fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # --- 2. EKSEN: Varyans (Secondary Axis) ---
    ax2 = ax1.twinx()

    valid_variances = [v for v in variances if v is not None]

    ax2.plot(valid_episodes, valid_variances, label='Variance', color='red', linewidth=2, alpha=0.5)
    ax2.set_ylabel("Variance", fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')


    plt.title("Q-Learning Stochastic Learning Curve", fontsize=16)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    ax1.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_learning_curve()