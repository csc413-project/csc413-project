import dm_control.suite as suite
from dm_control.suite.wrappers import pixels
import numpy as np

def get_random_action(action_spec):
    minimum, maximum = action_spec.minimum, action_spec.maximum
    shape = action_spec.shape
    return np.random.uniform(minimum, maximum, shape)


def main():
    env = suite.load(domain_name="acrobot", task_name="swingup")
    env = pixels.Wrapper(env)
    action_spec = env.action_spec()
    # collect 5 random episodes
    data = []
    time_step = env.reset()
    step_type, reward, discount, obs = time_step
    num_episodes = 0
    while num_episodes < 5:
        action = get_random_action(action_spec)
        time_step = env.step(action)
        step_type, reward, discount, next_obs = time_step 
        if time_step.last():
            step_type, reward, discount, next_obs = env.reset()
            num_episodes += 1
        else:
            data.append((obs, action, reward))
        obs = next_obs



if __name__ == "__main__":
    main()
