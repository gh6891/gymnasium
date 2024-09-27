import time
from mario import Mario



class Mario(Mario):
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model'])
        self.exploration_rate = checkpoint['exploration_rate']
        print(f"Loaded MarioNet from {checkpoint_path} with exploration rate {self.exploration_rate}")

# Load the trained model
checkpoint_path = "checkpoints/YOUR_CHECKPOINT_FILE.chkpt"
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
mario.load(checkpoint_path)

# Set exploration rate to 0 (greedy)
mario.exploration_rate = 0

# Run the trained Mario agent
state = env.reset()
done = False

while not done:
    # Render the game environment to see it visually
    env.render()

    # Mario agent takes an action
    action = mario.act(state)

    # Perform the action in the environment
    next_state, reward, done, trunc, info = env.step(action)

    # Update the state
    state = next_state

    # Slow down the loop to make it more watchable
    time.sleep(0.02)

env.close()