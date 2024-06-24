import numpy as np
import tensorflow as tf
from Environment import SatelliteEnv
import random
from Simulation import contour, run_simulation

# Parameters
num_satellites = 6
debris_speed = (0, 10)
initial_debris_position = (500, 100)
initial_positions = [[300, 300], [300, 400], [300, 500], [500, 300], [500, 400], [500, 500]]

# Initialize environment
env = SatelliteEnv(num_satellites=num_satellites, debris_speed=debris_speed, initial_positions=initial_positions, initial_debris_position=initial_debris_position)
state_size = env.state_space
action_size = env.action_space

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(state_size,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(action_size * num_satellites, activation='linear')
])
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Parameters for training
episodes = 50  # Adjusted for quick testing
timesteps = 80
epsilon = 0.7
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.95  # discount rate
batch_size = 5  # Adjusted for quick testing
memory = []

for e in range(episodes):
    print('episode:', e)
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(timesteps):  # Adjusted for quick testing
        if np.random.rand() <= epsilon:
            # print('random')
            actions = [np.random.randint(action_size) for _ in range(num_satellites)]
        else:
            # print('predict')
            q_values = model.predict(state, verbose=0)
            actions = [np.argmax(q_values[0][i * action_size: (i + 1) * action_size]) for i in range(num_satellites)]
        next_state, reward, done = env.step(actions, step=10)
        reward = reward if not done else -10
        # print(time, state, actions, reward)
        next_state = np.reshape(next_state, [1, state_size])
        # Store the experience in memory as a tuple
        memory.append((state, actions, reward, next_state, done))

        # Train the model using random minibatches from the memory
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for state_mb, actions_mb, reward_mb, next_state_mb, done_mb in minibatch:
                target = reward_mb
                if not done_mb:
                    target = (reward_mb + gamma * np.amax(model.predict(next_state_mb, verbose=0)[0]))
                target_f = model.predict(state_mb, verbose=0)
                for i, action in enumerate(actions_mb):
                    target_f[0][i * action_size + action] = target
                model.fit(state_mb, target_f, epochs=1, verbose=0)

        state = next_state
        if done:
            print(f"Episode: {e+1}/{episodes}, score: {time}, epsilon: {epsilon:.2}")
            break

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Save the model
model.save("satellite_model3_upperright.h5")

# tf.keras.saving.save_model(model, 'my_model.keras')

contour(lst_comm=range(50,200,30), lst_close=range(0, 150, 30), episodes=1)