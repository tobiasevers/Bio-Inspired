# Window parameters
WIDTH, HEIGHT = 800, 800  # Dimensions of the window in pixels
TIMESTEP = 20  # Time step for the simulation in milliseconds
UPDATERATE = 1  # Update rate for the simulation
NUM_SATELLITES = 6  # Number of satellites in the simulation

# Satellite parameters
COMMPERIMETER = 100  # Communication perimeter (distance within which satellites can communicate)
CLOSEPERIMETER = 50  # Close perimeter (distance within which satellites need to avoid debris)
VERYCLOSEPERIMETER = 10  # Very close perimeter (critical distance to avoid collisions)
DEBRISSIZE = 10  # Size of the debris
RANDOMSTD = 10  # Standard deviation for random movements (noise) in the simulation


