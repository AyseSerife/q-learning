import matplotlib.pyplot as plt
import csv


def plot_learning_curve(data_file="graph_data.txt"):
    episodes = []
    steps = []
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

                    # If moving average is not empty
                    if len(row) >= 3 and row[2] != "":
                        moving_averages.append(float(row[2]))
                    else:
                        moving_averages.append(None)  # Pending

    except FileNotFoundError:
        print(f"Error: File '{data_file}' doesn't exist.")
        return

    # Plot the graph
    plt.figure(figsize=(12, 6))

    # Plot raw steps
    plt.plot(episodes, steps, label='Raw Steps', color='lightgray', alpha=0.7)

    # Plot Moving Average
    valid_episodes = [e for e, m in zip(episodes, moving_averages) if m is not None]
    valid_moving_averages = [m for m in moving_averages if m is not None]

    plt.plot(valid_episodes, valid_moving_averages, label='Moving Average', color='blue',linewidth=2)

    plt.title("Q-Learning (Learning Curve)", fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Steps", fontsize=12)

    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_learning_curve()