#SoC-2025-Super-Mario-Quest-using-RL
By Divyaansh Narkhede Roll Number : 24B0981
# Reinforcement Learning Experiments

This repository contains a series of Jupyter notebooks documenting experiments in reinforcement learning, covering fundamental Q-learning and more advanced Proximal Policy Optimization (PPO) applied to environments.

## Project Structure and Overview

* `week_2.ipynb`: Implements Q-learning on the `Taxi-v3` environment.
* `week_3.ipynb`: Sets up the Super Mario Bros environment with basic preprocessing and demonstrates an error encountered during initial PPO model training.
* `week_4.ipynb`: Explores the Super Mario Bros environment and its observation space.
* `week_5.ipynb`: Demonstrates training PPO agents for Super Mario Bros with different movement sets (simple and complex) and across different game levels.

## Week-by-Week Breakdown

### Week 2: Q-Learning with Taxi-v3

The `week_2.ipynb` notebook focuses on implementing and evaluating a Q-learning agent for the `Taxi-v3` environment from OpenAI Gym.

* **Environment**: `Taxi-v3` is a classic reinforcement learning environment where an agent learns to navigate a taxi to pick up and drop off passengers.
* **Algorithm**: Q-learning, a value-based reinforcement learning algorithm, is used to train the agent.
* **Key Parameters**:
    * `alpha` (learning rate): 0.1
    * `gamma` (discount factor): 0.6
    * `epsilon` (exploration rate): 0.1
* **Training**: The agent is trained for 100,000 episodes.
* **Evaluation**: After training, the agent's performance is evaluated over 10,000 episodes, showing metrics like average timesteps, penalties, and rewards per episode.
    * Average timesteps per episode: 13.0837
    * Average penalties per episode: 0.0
    * Average rewards per episode: 0.6801

### Week 3: Super Mario Bros Environment Setup and Initial PPO Attempt

The `week_3.ipynb` notebook initiates the work with the Super Mario Bros environment.

* **Environment**: `SuperMarioBros-1-1-v0` is loaded and configured for simple movement.
* **Wrappers**:
    * `JoypadSpace`: Limits the action space to `SIMPLE_MOVEMENT`.
    * `GrayScaleObservation`: Converts observations to grayscale, keeping dimensions.
    * `DummyVecEnv`: Wraps the environment to be compatible with Stable Baselines3.
    * `VecFrameStack`: Stacks 4 frames to provide the agent with a sense of trajectory and memory.
* **Model**: A PPO (Proximal Policy Optimization) model with `CnnPolicy` is initialized for image processing.
* **Callback**: A custom `TrainAndLoggingCallback` is implemented to save the model every 10,000 steps.
* **Issue**: The notebook highlights a `TypeError` during model training (`reset() got an unexpected keyword argument 'seed'`), indicating a compatibility issue between the environment and the Stable Baselines3 `learn` method's `reset` call. This suggests a version mismatch or an incompatibility with how `gym_super_mario_bros` handles seeding compared to what Stable Baselines3 expects.

### Week 4: Exploring Super Mario Bros Environment

The `week_4.ipynb` notebook provides a basic interaction with the Super Mario Bros environment to understand its observation space and information dictionary.

* **Environment**: `SuperMarioBros-1-1-v0` is set up with `COMPLEX_MOVEMENT`.
* **Observation Space**: The notebook demonstrates how to reset the environment and print the `info` dictionary and the shape of the observation (state). This helps in understanding the kind of data the agent will receive (e.g., `'coins'`, `'life'`, `'score'`, `'x_pos'`, `'y_pos'` in the `info` dictionary, and the `(240, 256, 3)` shape for the RGB image observations).

### Week 5: Training PPO Models for Super Mario Bros

The `week_5.ipynb` notebook showcases successful training of PPO agents for Super Mario Bros using different movement sets and on a different level, building upon the environment setup.

* **Environment Preparation**: Similar to Week 3, the environment is wrapped with `JoypadSpace`, `ResizeObservation` (to 84x84), `GrayScaleObservation` (keeping dimensions), `Monitor` (for logging stats), `DummyVecEnv`, and `VecFrameStack` (n\_stack=4).

* **Simple Movement Training**:
    * **Movement Set**: `SIMPLE_MOVEMENT` is used.
    * **Training Time**: The model is trained for 100,000 total timesteps.
    * **Performance**: During training, the agent shows increasing `ep_rew_mean` (mean episode reward) and `ep_len_mean` (mean episode length), indicating learning progress. The final reported mean episode reward is 552, and the mean episode length is approximately 16,200 timesteps.
    * **Saving**: The trained model is saved to `train/ppo_mario_simple/mario_model`.

* **Complex Movement Training (Level 1-1)**:
    * **Movement Set**: `COMPLEX_MOVEMENT` is used.
    * **Training Time**: The model is trained for 100,000 total timesteps.
    * **Performance**: The agent shows improvement over time, with the final reported mean episode reward being 151 and the mean episode length being approximately 5,430 timesteps.
    * **Saving**: The trained model is saved to `train/ppo_mario_complex/mario_model`.

* **Complex Movement Training (Level 2-1)**:
    * **Environment**: The game level is changed to `SuperMarioBros-2-1-v0`.
    * **Movement Set**: `COMPLEX_MOVEMENT` is used.
    * **Hyperparameters**: Specific hyperparameters are set for this training run:
        * `learning_rate`: 2.5e-4
        * `n_steps`: 2048
        * `batch_size`: 64
        * `n_epochs`: 4
        * `ent_coef`: 0.01
        * `clip_range`: 0.1
    * **Training Time**: The model is trained for 1,000,000 total timesteps.
    * **Performance**: The training logs show a progressive increase in `ep_rew_mean` and a general decrease in `ep_len_mean` (though with fluctuations), suggesting the agent is learning to complete the level more efficiently. The final mean episode reward is 326, and the mean episode length is approximately 565 timesteps.
    * **Saving**: The trained model is saved to `train/ppo_mario_complex_2/mario_model`.

---
