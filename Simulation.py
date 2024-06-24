import numpy as np
import tensorflow as tf
from Parameters import WIDTH, HEIGHT, NUM_SATELLITES, COMMPERIMETER, CLOSEPERIMETER
from Environment import Environment
import pickle
from Plot import plot_heatmap


# Run a simulation with the specified parameters
def run_simulation(episodes=20, max_steps=800, speed=(0, 1), magspeed=1, initial_debris_position=(500, 0),
                   COMM=COMMPERIMETER, CLOSE=CLOSEPERIMETER, comm_ability=True, modelparam=None):
    # Define initial positions of the satellites
    initial_positions = [[300, 300], [300, 400], [300, 500], [500, 300], [500, 400], [500, 500]]
    debris_speed = tuple([x * magspeed for x in speed])
    results = []

    # Load the trained RL model
    model_path = f"Models/satellite_model3_upperright_epi{modelparam}.h5" if modelparam else "Models/satellite_model3_upperright.h5"
    model = tf.keras.models.load_model(model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

    state_size = 2 * NUM_SATELLITES + 2
    action_size = 5

    # Print simulation parameters
    print('Model:', model_path)
    print('COMM:', COMM, 'CLOSE:', CLOSE, 'CommAbility:', comm_ability, 'Speed:', debris_speed, 'Position:',
          initial_debris_position)

    lst_error = []
    lst_collision = []
    for episode in range(episodes):
        # Initialize the environment
        env = Environment(num_satellites=NUM_SATELLITES, debris_speed=debris_speed, initial_positions=initial_positions,
                          initial_debris_position=initial_debris_position, commper=COMM, closeper=CLOSE)
        state = np.reshape(env.reset(), [1, state_size])
        total_reward = 0

        error = 0
        collision = 0
        for step in range(max_steps):
            # Check for debris errors
            violation = any(sat.check_error_debris(env.debris) for sat in env.satellites)
            if violation:
                error += 1

            # Check for very close perimeter collisions
            violation2 = any(sat.check_verycloseperimeter(env.debris) for sat in env.satellites)
            if violation2:
                collision += 1

            # Scan debris and update knowledge
            for sat in env.satellites:
                sat.scan_debris(env.debris)

            # Communicate knowledge between satellites
            if comm_ability:
                for sat in env.satellites:
                    sat.communicate(env.satellites)

            # Check if any satellite sees the debris
            any_sees_debris = any(sat.check_commperimeter(env.debris) for sat in env.satellites)
            if any_sees_debris:
                state = np.array(
                    [coord for sat in env.satellites for coord in sat.position] + list(env.debris.position)).reshape(
                    (1, state_size))
                q_values = model.predict(state, verbose=0)
                actions = [np.argmax(q_values[0][i * action_size: (i + 1) * action_size]) if sat.see_debris else (
                    4 if sat.x < initial_positions[i][0] else
                    3 if sat.x > initial_positions[i][0] else
                    2 if sat.y < initial_positions[i][1] else
                    1 if sat.y > initial_positions[i][1] else 0) for i, sat in enumerate(env.satellites)]
            else:
                # Move satellites back to initial positions if no debris is seen
                actions = [4 if sat.x < initial_positions[i][0] else
                           3 if sat.x > initial_positions[i][0] else
                           2 if sat.y < initial_positions[i][1] else
                           1 if sat.y > initial_positions[i][1] else 0 for i, sat in enumerate(env.satellites)]

            # Update satellites based on actions
            for i, sat in enumerate(env.satellites):
                sat.move_based_on_action(actions[i], step=1)
            env.debris.move()

            state = np.reshape(env.get_state(), [1, state_size])

        lst_error.append(error)
        lst_collision.append(collision)
        results.append(total_reward)
        print('Errors and collisions:', error, collision)

    return results, lst_error, lst_collision


# Generate a heatmap for different communication and close perimeters
def contour(lst_comm, lst_close, episodes, modelparam=None):
    # Run simulations for each combination of communication and close perimeters
    dict_results = {comm: {close: sum(
        run_simulation(episodes=episodes, max_steps=800, COMM=comm, CLOSE=close, comm_ability=True,
                       modelparam=modelparam)[1]) for close in lst_close} for comm in lst_comm}

    comm_values = sorted(lst_comm)
    close_values = sorted(lst_close)
    Z = np.array([[dict_results[comm][close] for close in close_values] for comm in comm_values])

    # Plot the heatmap
    plot_heatmap(Z, f'Heatmap of {modelparam}', lst_close, lst_comm)
    return dict_results


# Compare the performance with and without communication for varying speeds
def speedandcomm(lst_speed, comm, close, episodes, debris_speed=(0, 1), debris_position=(500, 0)):
    dict_results = {'COMM': {speed: run_simulation(episodes=episodes, max_steps=800, speed=debris_speed, magspeed=speed,
                                                   initial_debris_position=debris_position, COMM=comm, CLOSE=close,
                                                   comm_ability=True)[1:3] for speed in lst_speed},
                    'NOCOMM': {
                        speed: run_simulation(episodes=episodes, max_steps=800, speed=debris_speed, magspeed=speed,
                                              initial_debris_position=debris_position, COMM=comm, CLOSE=close,
                                              comm_ability=False)[1:3] for speed in lst_speed}}
    return dict_results


# Analyze results for varying angles and speeds of debris
def varyangle(comm, close, lst_speed, lst_debris_speed, lst_debris_positions, lst_angles):
    return {angle: speedandcomm(lst_speed=lst_speed, comm=comm, close=close, episodes=1, debris_speed=debris_speed,
                                debris_position=debris_position) for angle, debris_speed, debris_position in
            zip(lst_angles, lst_debris_speed, lst_debris_positions)}


if __name__ == '__main__':
    lst_close = [50]
    lst_comm = range(50, 200, 10)
    lst_speed = [1, 2, 3, 4, 5]
    comm = 120
    close = 50

    # Run contour simulations and save results
    dict_results = contour(lst_comm=lst_comm, lst_close=lst_close, episodes=1, modelparam='50_2')
    with open(f'Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_50_2.pkl', 'wb') as file:
        pickle.dump(dict_results, file)

    # Run simulations for different episodes and save results
    lst_episodes = [25, 50, 75, 100]
    for episode in lst_episodes:
        dict_results = contour(lst_comm=lst_comm, lst_close=lst_close, episodes=1, modelparam=str(episode) + '_2')
        with open(f'Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_{episode}_2.pkl', 'wb') as file:
            pickle.dump(dict_results, file)

    # Define parameters for varying angles and speeds of debris
    lst_debris_speed = [(1 / 3, 3 / 6), (1 / 3, 1), (0, 1), (-1 / 3, 1), (-1 / 3, 3 / 6), (-1, 1), (-3 / 6, 1 / 3),
                        (-1, 1 / 3), (-1, 0)]
    lst_debris_positions = [(300, 0), (400, 0), (500, 0), (600, 0), (700, 0), (800, 0), (800, 100), (800, 200),
                            (800, 300)]
    lst_angles = ['-34deg', '-18deg', '0deg', '18deg', '34deg', '45deg', '56deg', '72deg', '90deg']

    # Run simulations for varying angles and speeds, then save results
    dict_results = varyangle(comm=comm, close=close, lst_speed=lst_speed, lst_debris_speed=lst_debris_speed,
                             lst_debris_positions=lst_debris_positions, lst_angles=lst_angles)
    print(dict_results)
    with open(f'Results/Contour_model3_120_50_angles_nocomm_upperright.pkl', 'wb') as file:
        pickle.dump(dict_results, file)
