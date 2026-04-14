import numpy as np
import math


class DeliveryEnvironment:
    def __init__(self, min_prob=0.70, max_prob=0.95):
        self.grid_size = 10
        self.start_pos = (0, 0)
        self.pickup_pos = (2, 4)
        self.dropoff_pos = (8, 9)
        self.truck_pos = self.start_pos
        self.has_load = False
        self.action_space = [0, 1, 2, 3]  # Up, Down, Left, Right
        self.transition_probs = {}
        self.zone_center = (8, 7)

        self.update_probabilities(min_prob, max_prob)

    def update_probabilities(self, min_prob, max_prob):
        self.transition_probs = {}

        # Find the maximum distance on the map
        corners = [(0, 0), (0, self.grid_size - 1), (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        max_dist = max(
            [math.sqrt((self.zone_center[0] - cx) ** 2 + (self.zone_center[1] - cy) ** 2) for cx, cy in corners])

        # To avoid a "Division by Zero" error if 1.0 is entered, we limit it to 0.999
        min_prob = min(0.999, max(0.1, min_prob))
        max_prob = min(0.999, max(0.1, max_prob))

        # Formula for converting percentage values from the interface into alpha weights
        a_min = (8 * min_prob) / (1 - min_prob)
        a_max = (8 * max_prob) / (1 - max_prob)

        # Linear increase (slope) factor based on distance
        slope = (a_max - a_min) / max_dist

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                dist = math.sqrt(pow(self.zone_center[0] - x, 2) + pow(self.zone_center[1] - y, 2))
                for i in range(len(self.action_space)):
                    alphas = np.ones(9)
                    a = slope * dist + a_min
                    alphas[i] = a
                    probs = np.random.dirichlet(alphas)
                    self.transition_probs[(x, y, i)] = probs

    def reset(self):
        self.truck_pos = self.start_pos
        self.has_load = False
        return self._get_state()

    def _get_state(self):
        return (self.truck_pos[0], self.truck_pos[1], int(self.has_load))

    def step(self, action):
        x, y = self.truck_pos
        reward = -1
        done = False

        coordinate_probs = self.transition_probs[(x, y, action)]
        actual_action = np.random.choice(9, p=coordinate_probs)

        if actual_action == 0 and x > 0:  # up
            x -= 1
        elif actual_action == 1 and x < self.grid_size - 1:  # down
            x += 1
        elif actual_action == 2 and y > 0:  # left
            y -= 1
        elif actual_action == 3 and y < self.grid_size - 1:  # right
            y += 1
        elif actual_action == 4 and x > 0 and y < self.grid_size - 1:  # up-right
            x -= 1
            y += 1
        elif actual_action == 5 and x < self.grid_size - 1 and y < self.grid_size - 1:  # down-right
            x += 1
            y += 1
        elif actual_action == 6 and x > 0 and y > 0:  # up-left
            x -= 1
            y -= 1
        elif actual_action == 7 and x < self.grid_size - 1 and y > 0:  # down-left
            x += 1
            y -= 1
        elif actual_action == 8:  # stay
            pass
        else:
            reward = -10

        self.truck_pos = (x, y)

        if self.truck_pos == self.pickup_pos and not self.has_load:
            self.has_load = True
            reward = 20
        elif self.truck_pos == self.dropoff_pos and self.has_load:
            reward = 100
            done = True

        return self._get_state(), reward, done, actual_action