import pygame
import numpy as np
import tensorflow as tf
from Satellite import Satellite
from Parameters import WIDTH, HEIGHT, TIMESTEP, NUM_SATELLITES, COMMPERIMETER, CLOSEPERIMETER
from Debris import Debris

# Initialize and run the simulation
def main():
    pygame.init()  # Initialize Pygame
    screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Set up the display
    clock = pygame.time.Clock()  # Create a clock object to manage the frame rate
    initial_positions = [[300, 300], [300, 400], [300, 500], [500, 300], [500, 400], [500, 500]]  # Initial positions for satellites
    satellites = [Satellite(i, initial_positions[i], COMMPERIMETER, CLOSEPERIMETER) for i in range(NUM_SATELLITES)]  # Initialize satellites

    debris = Debris((0, 5), (500, 0))  # Initialize debris

    # Load the trained RL model
    model = tf.keras.models.load_model("Models/satellite_model3_upperright.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    state_size = 2 * NUM_SATELLITES + 2  # Define the state size
    action_size = 5  # Define the number of possible actions

    simulationtime = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # Exit the loop if the window is closed

        screen.fill((0, 0, 0))  # Clear the screen

        # Scan for debris and update knowledge
        for sat in satellites:
            sat.scan_debris(debris)

        # Communicate knowledge between satellites
        for sat in satellites:
            sat.communicate(satellites)

        # Check if any satellite sees the debris
        any_sees_debris = any(sat.check_commperimeter(debris) for sat in satellites)

        if any_sees_debris:
            # Get the current state
            state = []
            for sat in satellites:
                state.extend(sat.position)
            state.extend(debris.position)
            state = np.array(state).reshape((1, state_size))

            # Use the RL model to predict the next action for each satellite that has knowledge
            q_values = model.predict(state, verbose=0)
            actions = []
            for i, sat in enumerate(satellites):
                if sat.see_debris:
                    action = np.argmax(q_values[0][i * action_size: (i + 1) * action_size])
                else:
                    if sat.x < initial_positions[i][0]:
                        action = 4  # Move right
                    elif sat.x > initial_positions[i][0]:
                        action = 3  # Move left
                    elif sat.y < initial_positions[i][1]:
                        action = 2  # Move down
                    elif sat.y > initial_positions[i][1]:
                        action = 1  # Move up
                    else:
                        action = 0  # Stay in place
                actions.append(action)
        else:
            # No satellite sees debris, move satellites back to initial positions
            actions = []
            for i, sat in enumerate(satellites):
                if sat.x < initial_positions[i][0]:
                    actions.append(4)  # Move right
                elif sat.x > initial_positions[i][0]:
                    actions.append(3)  # Move left
                elif sat.y < initial_positions[i][1]:
                    actions.append(2)  # Move down
                elif sat.y > initial_positions[i][1]:
                    actions.append(1)  # Move up
                else:
                    actions.append(0)  # Stay in place

        # Update satellites based on actions
        for i, sat in enumerate(satellites):
            sat.move_based_on_action(actions[i], step=1)

        # Update debris
        debris.move()

        # Draw satellites and debris
        for sat in satellites:
            sat.draw(screen)
        debris.draw(screen)

        pygame.display.flip()  # Update the full display
        clock.tick(TIMESTEP)  # Control the frame rate

        simulationtime += 1

    pygame.quit()  # Quit Pygame


if __name__ == "__main__":
    main()  # Run the main function
