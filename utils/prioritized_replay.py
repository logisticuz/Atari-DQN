import numpy as np
import random

class PrioritizedReplayBuffer:
    """Prioriterad Replay Buffer."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # 0 → uniform, 1 → full prioritisation
        self.pos = 0
        self.full = False

        # Förallokera lagring med fast storlek
        self.states      = np.zeros((capacity, 84, 84, 4), dtype=np.uint8)
        self.actions     = np.zeros(capacity, dtype=np.int32)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 84, 84, 4), dtype=np.uint8)
        self.dones       = np.zeros(capacity, dtype=bool)
        self.priorities  = np.zeros(capacity, dtype=np.float32)

    def add(self, experience, td_error: float = 1.0):
        state, action, reward, next_state, done = experience
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.priorities[self.pos] = max(self.priorities.max(), td_error)
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True
    # Samplar en från bufferten med prioritering
    def sample(self, batch_size: int, beta: float = 0.4):
        max_idx = self.capacity if self.full else self.pos
        if max_idx < batch_size:
            raise ValueError(f"Replay buffer has {max_idx} samples, need {batch_size}.")

        prios = self.priorities[:max_idx]
        probs = prios ** self.alpha
        if probs.sum() == 0 or np.any(~np.isfinite(probs)):
            probs = np.ones_like(prios, dtype=np.float64)
        probs = probs.astype(np.float64)

        # FP‑safe normalisering
        probs = np.clip(probs, 0.0, None)
        probs /= probs.sum()
        probs[-1] = 1.0 - probs[:-1].sum()  # exact 1.0

        indices = np.random.choice(max_idx, batch_size, replace=False, p=probs)
        weights = (max_idx * probs[indices]) ** (-beta)
        weights /= weights.max()

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices, td_errors):
        self.priorities[indices] = np.clip(np.abs(td_errors), 1e-6, 1e6)

    def save(self, path: str):
        np.savez_compressed(
            path,
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            next_states=self.next_states,
            dones=self.dones,
            priorities=self.priorities,
            pos=self.pos,
            full=self.full,
            capacity=self.capacity,
        )

    @classmethod
    def from_npz(cls, data):
        obj = cls(int(data["capacity"]))
        obj.states = data["states"]
        obj.actions = data["actions"]
        obj.rewards = data["rewards"]
        obj.next_states = data["next_states"]
        obj.dones = data["dones"]
        obj.priorities = data["priorities"]
        obj.pos = int(data["pos"])
        obj.full = bool(data["full"])
        return obj

    def __len__(self):
        return self.capacity if self.full else self.pos
