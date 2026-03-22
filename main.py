from environment import DeliveryEnvironment
from agent import QLearningAgent


def main():
    print("=== Q-LEARNING ===")

    try:
        episodes = int(input("Maximum Episode Num: "))
        epsilon_start = float(input("Starting Epsilon Value (0.0 - 1.0): "))
        epsilon_end = float(input("Final Epsilon Value: "))
    except ValueError:
        print("Invalid Input.")
        return

    env = DeliveryEnvironment()
    agent = QLearningAgent(env, episodes, epsilon_start, epsilon_end)

    agent.train()

    agent.test_and_save_solution(filename="solution_path.txt")


if __name__ == "__main__":
    main()