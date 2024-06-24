import pygame
import tensorflow as tf
from Satellite import Satellite
from Parameters import WIDTH, HEIGHT, TIMESTEP, NUM_SATELLITES, COMMPERIMETER, CLOSEPERIMETER
from Debris import Debris
import math

def draw_arrow(screen, color, start, end, arrow_width=5, arrow_length=10, arrow_angle=math.pi / 12):
    pygame.draw.line(screen, color, start, end, arrow_width)  # Draw the main line of the arrow

    # Calculate the angle of the arrow
    angle = math.atan2(end[1] - start[1], end[0] - start[0])

    # Calculate the points for the arrowhead
    arrow_point1 = (
        end[0] - arrow_length * math.cos(angle + arrow_angle),
        end[1] - arrow_length * math.sin(angle + arrow_angle)
    )
    arrow_point2 = (
        end[0] - arrow_length * math.cos(angle - arrow_angle),
        end[1] - arrow_length * math.sin(angle - arrow_angle)
    )

    # Draw the arrowhead
    pygame.draw.polygon(screen, color, [end, arrow_point1, arrow_point2])

def main():
    pygame.init()  # Initialize Pygame
    screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Create the window
    clock = pygame.time.Clock()  # Create a clock object to manage time
    initial_positions = [[300, 300], [300, 400], [300, 500], [500, 300], [500, 400], [500, 500]]  # Initial positions of satellites
    satellites = [Satellite(i, initial_positions[i], COMMPERIMETER, CLOSEPERIMETER) for i in range(NUM_SATELLITES)]  # Initialize satellites

    lst_debris = [Debris((0,1), (500, 50)), Debris((0,1), (300, 50)), Debris((0,1), (500, 750)), Debris((0,1), (300, 750)),
                  Debris((0,1), (50, 300)), Debris((0,1), (50, 500)), Debris((0,1), (750, 300)), Debris((0,1), (750, 500))]  # Initialize debris

    # Load the trained RL model
    model = tf.keras.models.load_model("Models/satellite_model3_upperright.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']  # Compile the model with appropriate settings
    )

    state_size = 2 * NUM_SATELLITES + 2  # State size for the model
    action_size = 5  # Number of possible actions

    simulationtime = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # Exit the loop if the window is closed

        screen.fill((0, 0, 0))  # Fill the screen with black

        # Draw satellites and debris
        for sat in satellites:
            sat.draw(screen)
        for debris in lst_debris:
            debris.draw(screen)

        # Draw arrows for visualization (commented examples)
        # draw_arrow(screen, 'red', (300, 700), (300,650), arrow_width=2, arrow_length=50)
        # draw_arrow(screen, 'red', (500, 700), (500, 650), arrow_width=2, arrow_length=50)
        # draw_arrow(screen, 'red', (300, 100), (300, 150), arrow_width=2, arrow_length=50)
        draw_arrow(screen, 'red', (500, 100), (500, 150), arrow_width=2, arrow_length=50)
        # draw_arrow(screen, 'red', (100, 300), (150, 300), arrow_width=2, arrow_length=50)
        # draw_arrow(screen, 'red', (100, 500), (150, 500), arrow_width=2, arrow_length=50)
        # draw_arrow(screen, 'red', (700, 300), (650, 300), arrow_width=2, arrow_length=50)
        # draw_arrow(screen, 'red', (700, 500), (650, 500), arrow_width=2, arrow_length=50)

        pygame.display.flip()  # Update the display
        clock.tick(TIMESTEP)  # Control the frame rate

        simulationtime += 1  # Increment the simulation time

    pygame.quit()  # Quit Pygame

if __name__ == "__main__":
    main()