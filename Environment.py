import numpy as np
from Satellite import Satellite
from Debris import Debris
import Parameters

# Class representing the environment with satellites and debris
class Environment:
    def __init__(self, num_satellites, debris_speed, initial_positions, initial_debris_position, commper=Parameters.COMMPERIMETER, closeper=Parameters.CLOSEPERIMETER):
        self.num_satellites = num_satellites  # Number of satellites
        self.debris = Debris(debris_speed, initial_debris_position)  # Initialize debris
        self.satellites = [Satellite(i, initial_positions[i], commper, closeper) for i in range(num_satellites)]  # Initialize satellites
        self.action_space = 5  # Number of possible actions: stay, up, down, left, right
        self.state_space = 2 * num_satellites + 2  # State space: positions of satellites and debris
        self.initial_positions = initial_positions  # Initial positions of satellites
        self.initial_debris_position = initial_debris_position  # Initial position of debris
        self.reset()  # Reset the environment
        self.commrange = commper  # Communication perimeter
        self.closerange = closeper  # Close perimeter

    def reset(self):
        # Reset the environment to the initial state
        for i, sat in enumerate(self.satellites):
            sat.reset()  # Reset each satellite
        self.debris.reset()  # Reset the debris
        return self.get_state()  # Return the initial state

    def get_state(self):
        # Return the current state of the environment
        state = []
        for sat in self.satellites:
            state.extend(sat.position)  # Add satellite positions to the state
        state.extend(self.debris.position)  # Add debris position to the state
        return np.array(state)  # Return the state as a numpy array

    def step(self, actions, step):
        # Execute the actions for each satellite and update the environment
        for i, action in enumerate(actions):
            self.satellites[i].move_based_on_action(action, step=step)  # Move satellites based on actions
        self.debris.move()  # Move the debris

        hit = self.check_done()  # Check if any satellite is in close perimeter with debris

        return self.get_state(), hit  # Return the new state and whether a satellite is hit

    def get_action_vector(self, action):
        # Convert the action into a movement vector.
        # Placeholder function, replace with actual action-to-movement conversion.
        action_vectors = {
            0: np.array([0, 0]),  # No movement
            1: np.array([1, 0]),  # Move right
            2: np.array([-1, 0]),  # Move left
            3: np.array([0, 1]),  # Move up
            4: np.array([0, -1])  # Move down
            # Add more actions if necessary
        }
        return action_vectors.get(action, np.array([0, 0]))  # Return the movement vector for the given action

    def check_done(self):
        # Check if the episode is done (e.g., debris in close perimeter)
        for sat in self.satellites:
            if sat.check_closeperimeter(self.debris):  # Check if any satellite is in close perimeter with debris
                return True  # Episode is done
        return False  # Episode is not done