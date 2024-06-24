Bio-Inspired Reinforcement Learning Model for Automatic Debris Avoidance
-
This code contains a foundation for an agent-based model using Reinforcement Learning and Swarm Intelligence. The code is developed for an assignment of the course _AE4350 Bio-Inspired Intelligence and Learning for Aerospace Applications_ at the Faculty of Aerospace Engineering at the Delft University of Technology in 2024.

<span style="color:blue; font-weight:bold;">**Use of the Code**</span>

Different RL models can be configured and trained in the 'train_outside.py' file. If the models configuration needs multiple training directions, use 'train_outside_multiple.py'. Make sure to change the name for savind at the bottom of the files. A Pygame simulation can be run to visualise the reaction of the satellite system using 'Sim_pygame' ('Pygame_env.py' is only used to create images for the environment setup). Make sure to define the following:

- Model used
- Start location debris
- Speed debris

To generate results, the 'Simulation.py' can be used. Pickle is used to save results dictionaries. Also make sure to define the model used.

Lastly, 'Plot.py' is used to plot certain results. In this model, already existing results dictionaries are loaded using pickle.

<span style="color:blue; font-weight:bold;">**Requirements**</span>

Run the following command in the terminal to acquire all necessary modules and packages:

_pip install -r requirements.txt_

<span style="color:blue; font-weight:bold;">**Remaining Python files**</span>

- Debris.py: contains debris agent framework
- Environment.py: contains environment with debris and satellites framework
- Parameters.py: contains parameters used in training and simulation
- Satellite.py: contains satellite agent framework

Remaining folders are explained below:

<span style="color:blue; font-weight:bold;">**Models**</span>

Already multiple configurations of the model have been created and stored in the 'Models' folder. To explain the naming:
- 'model3': last version of the model
- 'multiple': multiple directions used during training
- 'vertical': trained in vertical direction
- 'horizontal': trained in horizontal directions
- 'upperright': this refers to the column which is trained
- '_025': refers to epsilon used
- 'epi25': refers to the number of episodes used for training

<span style="color:blue; font-weight:bold;">**Results**</span>

Already some results have been generated by using the different model configurations explained above. To explain the naming:

- 'Contour_model3': last version of model used 
- '50_200': refers to the communication range. This can be one number of only 1 communication range was used. 
- '0_150': refers to the close range. This can also be one number.
- 'comm/nocomm': defines if the communication ability was activated.