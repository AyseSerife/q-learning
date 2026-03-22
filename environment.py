import numpy as np


class DeliveryEnvironment:
    def __init__(self):
        self.grid_size = 10
        self.start_pos = (0, 0)
        self.pickup_pos = (2, 4)
        self.dropoff_pos = (8, 9)
        self.truck_pos = self.start_pos
        self.has_load = False
        self.action_space = [0, 1, 2, 3] # Up, Down, Left, Right
        self.transition_probs = {}
        self.zone_center = (8,7)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                dist = abs(self.zone_center[0] - x) + abs(self.zone_center[1] - y)
                for i in range(len(self.action_space)):
                    alphas = np.ones(9)
                    a = 1.5 * dist + 4
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
        action = np.random.choice(9, p=coordinate_probs)

        if action == 0 and x > 0: # up
            x -= 1
        elif action == 1 and x < self.grid_size - 1: # down
            x += 1
        elif action == 2 and y > 0: # left
            y -= 1
        elif action == 3 and y < self.grid_size - 1: # right
            y += 1
        elif action == 4 and x > 0 and y < self.grid_size - 1: # up-right
            x -= 1
            y += 1
        elif action == 5 and x < self.grid_size - 1 and y < self.grid_size - 1: # down-right
            x += 1
            y += 1
        elif action == 6 and x > 0 and y > 0: # up-left
            x -= 1
            y -= 1
        elif action == 7 and x < self.grid_size - 1 and y > 0: # down-left
            x += 1
            y -= 1
        elif action == 8: # stay
            pass
        else:
            reward = -10

        self.truck_pos = (x, y)

        if self.truck_pos == self.pickup_pos and not self.has_load: # load
            self.has_load = True
            reward = 20
        elif self.truck_pos == self.dropoff_pos and self.has_load: # unload
            reward = 100
            done = True

        return self._get_state(), reward, done