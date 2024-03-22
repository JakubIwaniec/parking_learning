# Gymnasium - Parking Car Environment

This environment simulates a parking car scenario where the goal is to park a car at a designated destination. The car can accelerate, brake, and rotate to navigate towards the destination.

## Installation

[//]: # (You can install Gymnasium along with this environment using pip:)

[//]: # (```bash)

[//]: # (pip install gymnasium[classic-control])

[//]: # (```)

[//]: # (## Usage)

[//]: # ()
[//]: # (```python)

[//]: # (import gymnasium as gym)

[//]: # ()
[//]: # ()
[//]: # (# Create an instance of the Parking Car environment)

[//]: # (env = gym.make&#40;'ParkingCar-v0'&#41;)

[//]: # ()
[//]: # (# Reset the environment)

[//]: # (obs = env.reset&#40;&#41;)

[//]: # ()
[//]: # (done = False)

[//]: # (while not done:)

[//]: # (    # Perform a random action)

[//]: # (    action = env.action_space.sample&#40;&#41;)

[//]: # (    )
[//]: # (    # Step through the environment)

[//]: # (    obs, reward, done, _ = env.step&#40;action&#41;)

[//]: # (    )
[//]: # (    # Render the environment &#40;optional&#41;)

[//]: # (    env.render&#40;&#41;)

[//]: # ()
[//]: # (# Close the environment)

[//]: # (env.close&#40;&#41;)

[//]: # (```)

## Environment Details

### Observation Space

The observation space consists of a 6-element array:

-  X-coordinate of the car
-  Y-coordinate of the car
-  Velocity of the car
-  Rotation angle of the car
-  X-coordinate of the destination
-  Y-coordinate of the destination

### Action Space

The action space consists of 5 discrete actions:

- `0`: Do nothing
- `1`: Gas (accelerate)
- `2`: Brake
- `3`: Rotate left
- `4`: Rotate right

### Rewards

The agent receives a reward of `0` at each time step.

### Termination

The episode terminates when the car goes out of bounds or reaches the destination.

## Rendering

This environment supports rendering in two modes:

- **human**: Renders the environment using Pygame for visualization.
- **rgb_array**: Returns the rendered environment as an RGB array.

## Author

This environment is authored by Miłosz Stolarski, Jakub Iwaniec and Miłosz Gostyński. 

For more details and contributions, refer to the [Gymnasium Documentation](https://gymnasium.farama.org/).

## License

This environment is released under the [MIT License](https://opensource.org/licenses/MIT).