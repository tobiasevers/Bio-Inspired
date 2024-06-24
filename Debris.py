from Parameters import WIDTH, HEIGHT
import pygame

# Class to represent a piece of debris in the game
class Debris():

    # Initialize the debris with speed and initial position
    def __init__(self, debris_speed, initial_position):
        self.initial_position = initial_position  # Save the initial position
        self.x = initial_position[0]  # Set initial x-coordinate
        self.y = initial_position[1]  # Set initial y-coordinate
        self.position = (self.x, self.y)  # Current position tuple
        self.vx = debris_speed[0]  # Velocity in x-direction
        self.vy = debris_speed[1]  # Velocity in y-direction
        self.velocity = (self.vx, self.vy)  # Current velocity tuple

    # Reset the debris to its initial position
    def reset(self):
        self.x = self.initial_position[0]  # Reset x-coordinate
        self.y = self.initial_position[1]  # Reset y-coordinate
        self.position = self.initial_position  # Update position tuple

    # Move the debris according to its velocity
    def move(self):
        self.x += self.vx  # Update x-coordinate based on velocity
        self.y += self.vy  # Update y-coordinate based on velocity
        self.position = (self.x, self.y)  # Update position tuple
        # self.wrap_around()  # Uncomment if wrap around behavior is needed

    # Handle wrap around behavior for debris (uncomment if needed)
    def wrap_around(self):
        if self.x < 0: self.x += WIDTH  # Wrap around left edge
        if self.x > WIDTH: self.x -= WIDTH  # Wrap around right edge
        if self.y < 0: self.y += HEIGHT  # Wrap around top edge
        if self.y > HEIGHT: self.y -= HEIGHT  # Wrap around bottom edge

    # Draw the debris on the screen
    def draw(self, screen):
        # Draw a red circle representing the debris
        pygame.draw.circle(screen, 'red', (self.x, self.y), 5)