# SoC-2025-Super-Mario-Quest-using-RL
Introduction
This project is all about teaching computers to play games—specifically, classic games like Taxi-v3 and Super Mario Bros—using the magic of reinforcement learning (RL). The journey covers everything from simple grid navigation to the pixelated adventures of Mario, showing how AI can learn, stumble, and eventually succeed at tasks that once seemed only possible for humans.

Part 1: Taxi-v3 – Learning the Basics
We started with the Taxi-v3 environment, a simple grid world where the goal is to pick up and drop off passengers efficiently. This is a classic playground for RL beginners.

How it works:

The agent (our "taxi") explores the environment, trying different actions (move up, down, pick up, drop off, etc.).

We used a Q-learning algorithm, which means the agent updates a big table of "what's the best thing to do in this situation?" based on its experiences.

The more the agent plays, the better it gets at choosing actions that lead to success.

Results:

After 100,000 training episodes, the taxi agent became impressively efficient.

On average, it completed its task in about 13 steps, with almost no penalties and a solid average reward.

A peek at the code:

python
for i in range(1, 100001):
    state, _ = env.reset()
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        next_state, reward, done, _, _ = env.step(action)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        state = next_state
        
Part 2: Super Mario Bros – Leveling Up
After mastering Taxi-v3, we took on a much bigger challenge: teaching an AI to play Super Mario Bros. This is a whole new ballgame—Mario's world is visually complex, and the possible actions are much richer.

How we set it up:

We used the gym_super_mario_bros environment, which simulates the classic NES game.

To simplify things for the AI, we limited the action space (for example, "move right," "jump," etc.).

We preprocessed the game visuals—converting them to grayscale and stacking frames—so the AI could "see" and remember what's happening.

We trained a deep RL agent using PPO (Proximal Policy Optimization), a popular algorithm for learning from images.

Training the AI:

The model learned by playing millions of steps, slowly figuring out how to survive longer and collect more points.

We set up a callback function to save the model regularly, so we could track progress and avoid losing everything to a crash.

Challenges along the way:

Compatibility issues between different versions of Gym and wrappers caused some frustrating errors (like reset() got an unexpected keyword argument 'seed'). These required careful debugging and sometimes custom wrappers to fix.

Training took a lot of time and computing power—Mario is a tough teacher.

What Did We Learn?
RL is powerful, but not magic: Even simple environments require lots of trial and error for the agent to learn good strategies.

Deep RL for games is challenging: Mario is much harder than Taxi-v3. Training takes time, and getting the environment setup right is half the battle.

Debugging is part of the process: Version mismatches and API changes can break things in unexpected ways. Patience and persistence pay off.

Final Thoughts
This project was a fun and rewarding dive into the world of AI and games. Watching an agent go from clueless to competent—whether it's picking up passengers or dodging Goombas—never gets old. There’s still a lot to explore, especially in fine-tuning deep RL for complex games like Mario, but the foundation is strong.