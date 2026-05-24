# CartPole-v1 environment reference

This project uses [Gymnasium](https://gymnasium.farama.org/environments/classic_control/cart_pole/) (`gymnasium.make("CartPole-v1")`), which implements the same classic cart-pole task as the original OpenAI Gym environment.

The description below is adapted from the [OpenAI Gym `cartpole.py` docstring](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py). Terminology matches Gymnasium: `terminated` = failure below, `truncated` = max episode length.

## Description

This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

## Action space

The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction of the fixed force the cart is pushed with.

| Num | Action                 |
|-----|------------------------|
| 0   | Push cart to the left  |
| 1   | Push cart to the right |

**Note:** The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it.

## Observation space

The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

| Num | Observation           | Min                 | Max               |
|-----|-----------------------|---------------------|-------------------|
| 0   | Cart Position         | -4.8                | 4.8               |
| 1   | Cart Velocity         | -Inf                | Inf               |
| 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
| 3   | Pole Angular Velocity | -Inf                | Inf               |

**Note:** While the ranges above denote the possible values for observation space of each element, it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:

- The cart x-position (index 0) can take values between `(-4.8, 4.8)`, but the episode **terminates** if the cart leaves the `(-2.4, 2.4)` range.
- The pole angle can be observed between `(-.418, .418)` radians (or **±24°**), but the episode **terminates** if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**).

In code (`explore.py`, `train.py`, `play.py`):

```python
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

## Rewards

Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken, including the termination step, is allotted. The threshold for “solved” on v1 is an average reward of **475** over 100 consecutive episodes (i.e. holding for ~475 steps on average).

## Starting state

All observations are assigned a uniformly random value in `(-0.05, 0.05)`.

## Episode end

The episode ends if any one of the following occurs:

1. **Termination:** Pole angle is greater than ±12°
2. **Termination:** Cart position is greater than ±2.4 (center of the cart reaches the edge of the display)
3. **Truncation:** Episode length is greater than **500** (200 for v0)

## Usage in this repo

```python
import gymnasium as gym

env = gym.make("CartPole-v1")                    # training (no window)
env = gym.make("CartPole-v1", render_mode="human")  # explore / play
```

No additional environment arguments are required for the scripts here.

## Scripts in this folder

| Script       | Purpose                          |
|--------------|----------------------------------|
| `explore.py` | Random agent, UI with re-run     |
| `train.py`   | PPO training (SB3)               |
| `play.py`    | Run a saved PPO model            |

## Further reading

- [Gymnasium CartPole docs](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [OpenAI Gym source (original docstring)](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)
