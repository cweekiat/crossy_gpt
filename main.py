import openai
import gym
import re
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2

# Initialize the GPT-3 API
openai.api_key = ''
batch_size = 32
episodes = 100

# Initialize lists to store the number of actions and scores
action_counts = []
scores = []

# Game state preprocessing
def preprocess(observation):

    if isinstance(observation, tuple):  # Check if observation is a tuple
        # Assuming the first element of the tuple is the image array
        image_array = observation[0]
    else:
        image_array = observation

    # resize to two dimensions and makes grayscale
    observation = cv2.cvtColor(cv2.resize(image_array, (108, 118)), cv2.COLOR_BGR2GRAY)
    # crop image (top is just the score and bottom is activision logo)
    observation = observation[9:109,8:108]
    # reshape to 100x100x1 matrix
    return np.reshape(observation,(100,100,1))

# Function to generate action using GPT-3
def act(state):
    start_time = time.time()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are playing an Atari Freeway game. \
                                            The state of the game is represented by a 100x100x1 image. \
                                            In the action space, 0 means no operation, 1 means move up, and 2 means move down. \
                                            The goal is to help the chicken cross the road (top of the image) as fast as possible. \
                                            You will be send backwards (down) if you are hit by a car.\
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

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter('freeway_video.mp4',fourcc, 30.0, (160,210))

# Training phase
if __name__ == "__main__":
    ENV_NAME = "ALE/Freeway-v5"
    # Set up one instance of Freeway game environment
    env = gym.make(ENV_NAME, render_mode='human', frameskip=(2, 5), max_episode_steps=250, mode=0)
    # initialize seed
    env.seed(123)

    # ## To observe how preprocessing works on the first game frame ###
    # state = env.reset()
    # action0 = 0  # do nothing
    # observation0, reward0, terminal,truncated, info = env.step(action0)
    # print("Before processing: " + str(np.array(observation0).shape))
    # plt.imshow(np.array(observation0))
    # plt.show()
    # observation0 = preprocess(observation0)
    # print("After processing: " + str(np.array(observation0).shape))
    # plt.imshow(np.array(np.squeeze(observation0)))
    # plt.show()
    try:

        # perform training for episodes
        for e in range(episodes):
            # reset game state at beginning of each game
            state = env.reset()
            # preprocess the game state
            state = preprocess(state)

            # game is not done until time ends
            done = False
            truncated = False

            # current episode's game score
            curr_score = 0

            action_count = 0

            while not done and not truncated:
                frame = env.render()  # Specify the render mode here

                # decide on an action
                action = act(state)
                action_count += 1  

                # Ensure the action is within the valid range
                action = max(0, min(action, env.action_space.n - 1))

                # update the game based off of action
                next_state, reward, done, truncated, info = env.step(action)

                # Write the frame to the video
                video_out.write(cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR))

                if done or truncated:
                    print('Time ran out!')

                # reward is only one if chicken crossed the road, 0 otherwise
                next_state = preprocess(next_state)
                curr_score += reward

                # update state
                state = next_state

                if reward == 1:
                    print("Chicken crossed the road!")
                    print(action_count)
                    

            # once the game finishes (time runs out), output the game score
            print("episode: {}/{}, score: {}".format(e, episodes, curr_score))

            # Log the number of actions and score
            action_counts.append(action_count)
            scores.append(curr_score)

    except KeyboardInterrupt:
        print("Ctrl+C pressed. Exiting...")
    
    # Release the video writer
    video_out.release()

    # Plot the number of actions and scores
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(action_counts, label='Number of Actions', marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Number of Actions')
    plt.title('GPT - Number of Actions per Episode')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(scores, label='Score', marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('GPT - Score per Episode')
    plt.legend()

    plt.tight_layout()
    plt.show()
