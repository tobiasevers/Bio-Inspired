import numpy as np
import random as rd
from Parameters import WIDTH, HEIGHT, COMMPERIMETER, CLOSEPERIMETER, VERYCLOSEPERIMETER, RANDOMSTD
import pygame

class Satellite:
    def __init__(self, id, initial_position, commper, closeper, vx=0, vy=0):
        self.id = id  # Satellite identifier
        self.color = 'blue'  # Color of the satellite for drawing
        self.initial_position = initial_position  # Initial position of the satellite
        self.x = initial_position[0]  # x-coordinate of the satellite
        self.y = initial_position[1]  # y-coordinate of the satellite
        self.position = (self.x, self.y)  # Current position tuple
        self.vx = vx  # Velocity in x-direction
        self.vy = vy  # Velocity in y-direction
        self.commrange = commper  # Communication perimeter
        self.closerange = closeper  # Close perimeter
        self.see_debris = False  # Whether the satellite sees debris
        self.fact_debris = [None, None, None, None, None]  # Information about debris
        self.time = 0  # Time step counter

    def reset(self):
        # Reset position and other states
        self.position = self.initial_position
        self.x = self.initial_position[0]
        self.y = self.initial_position[1]

    def check_commperimeter(self, obj):
        # Check if the object is within the communication perimeter
        return np.sqrt((self.x - obj.x) ** 2 + (self.y - obj.y) ** 2) <= self.commrange

    def check_closeperimeter(self, obj):
        # Check if the object is within the close perimeter
        return np.sqrt((self.x - obj.x) ** 2 + (self.y - obj.y) ** 2) <= self.closerange

    def check_verycloseperimeter(self, obj):
        # Check if the object is within the very close perimeter
        return np.sqrt((self.x - obj.x) ** 2 + (self.y - obj.y) ** 2) <= VERYCLOSEPERIMETER

    def move(self):
        # Move the satellite based on its velocity
        self.x += self.vx
        self.y += self.vy
        self.position = (self.x, self.y)
        self.wrap_around()

    def move_based_on_action(self, action, step):
        # Define actions: 0 = stay, 1 = up, 2 = down, 3 = left, 4 = right
        if action == 1:
            self.y -= step  # Move up
        elif action == 2:
            self.y += step  # Move down
        elif action == 3:
            self.x -= step  # Move left
        elif action == 4:
            self.x += step  # Move right
        self.position = (self.x, self.y)
        self.wrap_around()

    def check_dist_debris(self):
        # Check the distance to the debris
        if self.fact_debris[0] is None:
            return False
        else:
            return np.sqrt((self.x - self.fact_debris[0]) ** 2 + (self.y - self.fact_debris[1]) ** 2) <= COMMPERIMETER

    def check_error_debris(self, debris):
        # Check if the satellite is too close to the debris
        dist = np.sqrt((self.x - debris.x) ** 2 + (self.y - debris.y) ** 2)
        return dist < self.closerange

    def update_knowledge(self, x_debris, y_debris, vx_debris, vy_debris, time):
        # Update the satellite's knowledge about the debris
        self.fact_debris = [x_debris, y_debris, vx_debris, vy_debris, time]
        self.see_debris = True

    def scan_debris(self, debris):
        # Scan for debris and update knowledge
        if self.check_commperimeter(debris):
            perc_x_debris = debris.x + rd.gauss(0, RANDOMSTD)
            perc_y_debris = debris.y + rd.gauss(0, RANDOMSTD)
            perc_vx_debris = debris.vx + rd.gauss(0, RANDOMSTD) if debris.vx != 0 else 0
            perc_vy_debris = debris.vy + rd.gauss(0, RANDOMSTD) if debris.vy != 0 else 0
            self.update_knowledge(perc_x_debris, perc_y_debris, perc_vx_debris, perc_vy_debris, self.time)
            return True
        else:
            return False

    def communicate(self, satellites):
        # Communicate knowledge with other satellites
        for sat in satellites:
            if self.check_commperimeter(sat):
                if self.fact_debris[4] is not None or sat.fact_debris[4] is not None:
                    if self.fact_debris[4] is not None and sat.fact_debris[4] is None:
                        sat.update_knowledge(self.fact_debris[0], self.fact_debris[1], self.fact_debris[2], self.fact_debris[3], self.fact_debris[4])
                    elif sat.fact_debris[4] is not None and self.fact_debris[4] is None:
                        self.update_knowledge(sat.fact_debris[0], sat.fact_debris[1], sat.fact_debris[2], sat.fact_debris[3], sat.fact_debris[4])
                    if self.fact_debris[4] is not None and sat.fact_debris[4] is not None:
                        if self.fact_debris[4] > sat.fact_debris[4]: # If own information is newer than older satellite's information
                            sat.update_knowledge(self.fact_debris[0], self.fact_debris[1], self.fact_debris[2], self.fact_debris[3], self.fact_debris[4])
                        else:
                            self.update_knowledge(sat.fact_debris[0], sat.fact_debris[1], sat.fact_debris[2], sat.fact_debris[3], sat.fact_debris[4])

    def wrap_around(self):
        # Wrap around the screen edges
        if self.x < 0: self.x += WIDTH
        if self.x > WIDTH: self.x -= WIDTH
        if self.y < 0: self.y += HEIGHT
        if self.y > HEIGHT: self.y -= HEIGHT

    def timestep(self, timestep):
        # Increment the time step
        self.time += timestep

    def draw(self, screen):
        # Draw the satellite and its ranges on the screen
        pygame.draw.circle(screen, self.color, (self.x, self.y), 5)
        pygame.draw.circle(screen, 'red', (self.x, self.y), CLOSEPERIMETER, 1)
        pygame.draw.circle(screen, 'orange', (self.x, self.y), COMMPERIMETER, 1)

    def __str__(self):
        # String representation of the satellite
        return f'Satellite {self.id} at position {self.x}, {self.y}'
