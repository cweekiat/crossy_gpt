import openai
import gym
import re
import preprocess as pp
import matplotlib.pyplot as plt
import numpy as np
import time
# Initialize the GPT-3 API
openai.api_key = 'insert ur api key'

# Function to generate action using GPT-3
def generate_action(state):
    start_time = time.time()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are playing an Atari Freeway game. \
                                            The state of the game is represented by a 100x100x1 image. \
                                            In the action space, 0 means no operation, 1 means move up, and 2 means move down. \
                                            The goal is to help the chicken cross the road (top of the image) as fast as possible. \
                                            Only return an integer (0, 1, or 2) as the output."},
            {"role": "user", "content": f"Read the current state of the game: {state}, output an integer, either 0, 1, or 2 for the chicken to cross the road."}
        ],
        max_tokens=20  # Reduced max tokens for a more concise output
    )

    action_match = re.search(r'\b\d+\b', response["choices"][0]["message"]["content"])
    if action_match:
        action = int(action_match.group())
    else:
        action = 0  # Default action if no valid action is found

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Action: {action}, Execution time: {execution_time} seconds")

    return action



def act(state):
    # Use GPT-3 to generate the action
    action = generate_action(state)
    # Convert action to integer if it's a string
    if isinstance(action, str):
        action = int(action)
    return action

# Training phase
if __name__ == "__main__":
    ENV_NAME = "ALE/Freeway-v5"
    # Set up one instance of Freeway game environment
    env = gym.make(ENV_NAME, render_mode='human')
    # initialize seed
    env.seed(123)
    batch_size = 32
    episodes = 10
    state = env.reset()
    action0 = 0  # do nothing
    observation0, reward0, terminal,truncated, info = env.step(action0)
    print("Before processing: " + str(np.array(observation0).shape))
    plt.imshow(np.array(observation0))
    plt.show()
    observation0 = pp.preprocess(observation0)
    print("After processing: " + str(np.array(observation0).shape))
    plt.imshow(np.array(np.squeeze(observation0)))
    plt.show()

    # perform training for episodes
    for e in range(episodes):
        # reset game state at beginning of each game
        print(e)
        state = env.reset()
        # preprocess the game state
        state = pp.preprocess(state)

        # game is not done until time ends
        done = False

        # current episode's game score
        curr_score = 0

        while not done:
            env.render()  # Specify the render mode here
            # decide on an action
            action = act(state)

            # Ensure the action is within the valid range
            action = max(0, min(action, env.action_space.n - 1))

            # update the game based off of action
            next_state, reward, done, truncated, info = env.step(action)
            # reward is only one if chicken crossed the road, 0 otherwise
            next_state = pp.preprocess(next_state)
            curr_score += reward

            # update state
            state = next_state

        # once the game finishes (time runs out), output the game score
        print("episode: {}/{}, score: {}".format(e, episodes, curr_score))
