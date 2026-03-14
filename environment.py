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

        if action == 0 and x > 0: # up
            x -= 1
        elif action == 1 and x < self.grid_size - 1: # down
            x += 1
        elif action == 2 and y > 0: # left
            y -= 1
        elif action == 3 and y < self.grid_size - 1: # right
            y += 1
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