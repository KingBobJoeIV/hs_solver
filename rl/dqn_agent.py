import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
else:
    device = torch.device("cpu")
    print("MPS device not available, falling back to CPU.")

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def push(self, *args):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.pos] = Transition(*args)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]
        probs = prios**self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        device=None,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        prioritized_replay=False,
        alpha=0.6,
        beta=0.4,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.steps_done = 0
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.prioritized_replay = prioritized_replay
        self.beta = beta
        if prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size, alpha)
        else:
            self.memory = ReplayBuffer(buffer_size)

    def select_action(self, state, legal_action_idxs):
        sample = random.random()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if sample < self.epsilon:
            return random.choice(legal_action_idxs)
        with torch.no_grad():
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.policy_net(state)
            q_values = q_values[0, legal_action_idxs]
            max_idx = torch.argmax(q_values).item()
            return legal_action_idxs[max_idx]

    def store_transition(self, *args):
        self.memory.push(*args)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        if self.prioritized_replay:
            transitions, indices, weights = self.memory.sample(
                self.batch_size, self.beta
            )
            batch = Transition(*zip(*transitions))
        else:
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            weights = np.ones(self.batch_size, dtype=np.float32)
            indices = None
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        td_errors = (q_values - target).abs().detach().cpu().numpy()
        loss = (weights * (q_values - target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.prioritized_replay and indices is not None:
            self.memory.update_priorities(indices, td_errors)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
