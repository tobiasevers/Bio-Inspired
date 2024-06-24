import numpy as np
import tensorflow as tf
from Environment import Environment
import random
from Parameters import CLOSEPERIMETER, WIDTH
from Simulation import contour, run_simulation

# Check if any satellite is within the CLOSEPERIMETER of the debris
def check_reward_states(states):
    for state in states:
        state = state[0]
        lst_sat = [(state[0], state[1]), (state[2], state[3]), (state[4], state[5]), (state[6], state[7]), (state[8], state[9]), (state[10], state[11])]
        debris = (state[12], state[13])
        succes = True
        for sat in lst_sat:
            if np.sqrt((sat[0] - debris[0]) ** 2 + (sat[1] - debris[1]) ** 2) <= CLOSEPERIMETER:
                succes = False
    return succes

# Calculate the average distance of satellites from their initial positions
def check_reward_dist(states, initial_positions):
    state = states[-1][0]
    lst_sat = [(state[0], state[1]), (state[2], state[3]), (state[4], state[5]), (state[6], state[7]), (state[8], state[9]), (state[10], state[11])]
    distcoef = 0
    for i, sat in enumerate(lst_sat):
        dist = np.linalg.norm(np.array(sat) - np.array(initial_positions[i]))
        distcoef += dist / WIDTH
    return distcoef / 6

# Parameters
num_satellites = 6
debris_speed = [(0, 10), (0, 10)]
initial_debris_position = [(500, 0), (300, 0)]
initial_positions = [[300, 300], [300, 400], [300, 500], [500, 300], [500, 400], [500, 500]]

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(14,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(5 * num_satellites, activation='linear')
])
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Parameters for training
episodes = 100  # Number of training episodes
timesteps = 80  # Number of timesteps per episode
epsilon = 0.8  # Exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.98  # Decay rate for exploration
gamma = 0.95  # Discount rate for future rewards
batch_size = 10  # Batch size for training
memory = []  # Memory to store experiences

for e in range(episodes):
    # Determine the mode based on the episode number
    if e / episodes < 0.50:
        mode = 0
    elif 0.50 <= e / episodes < 1.0:
        mode = 1
    print(f'Training ... episode: {e + 1}/{episodes} with debris start location: {initial_debris_position[mode]} and debris speed: {debris_speed[mode]}')

    # Initialize environment
    env = Environment(num_satellites=num_satellites, debris_speed=debris_speed[mode], initial_positions=initial_positions, initial_debris_position=initial_debris_position[mode])
    state_size = env.state_space
    action_size = env.action_space

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    lst_state = []
    lst_nstate = []
    lst_actions = []
    once_hit = False
    for time in range(timesteps):
        if np.random.rand() <= epsilon:
            actions = [np.random.randint(action_size) for _ in range(num_satellites)]
        else:
            q_values = model.predict(state, verbose=0)
            actions = [np.argmax(q_values[0][i * action_size: (i + 1) * action_size]) for i in range(num_satellites)]
        next_state, hit = env.step(actions, step=10)
        next_state = np.reshape(next_state, [1, state_size])

        lst_state.append(state)
        lst_nstate.append(next_state)
        lst_actions.append(actions)

        state = next_state
        if hit:
            once_hit = True

    global_reward = 0
    succescount = 0
    if once_hit:
        global_reward -= 10
    else:
        succescount += 1
        global_reward += 20

    coef = check_reward_dist(lst_state, initial_positions)
    if coef < 0.30:
        global_reward += 20
    else:
        global_reward -= 20

    for i, state in enumerate(lst_state):
        memory.append((lst_state[i], lst_actions[i], global_reward, lst_nstate))

    # Train the model using random minibatches from the memory
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
        for state_mb, actions_mb, reward_mb, next_state_mb in minibatch:
            target = reward_mb + gamma * np.amax(model.predict(next_state_mb, verbose=0)[0])
            target_f = model.predict(state_mb, verbose=0)
            for i, action in enumerate(actions_mb):
                target_f[0][i * action_size + action] = target
            model.fit(state_mb, target_f, epochs=1, verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Save the model
model.save("satellite_model3_multiple_upperright_moredistreward.h5")

print('Total success count:', succescount)
