import torch


class UpesiReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, theta_dim, device='cpu'):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.device = device

        self.obs = torch.zeros((self.capacity, obs_dim), dtype=torch.float, device=self.device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float, device=self.device)
        self.next_obs = torch.zeros((self.capacity, obs_dim), dtype=torch.float, device=self.device)
        self.theta_norm = torch.zeros((self.capacity, theta_dim), dtype=torch.float, device=self.device)

        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add_batch(self, obs, actions, next_obs, theta_norm):
        batch_size = int(obs.shape[0])
        if batch_size == 0:
            return

        obs = obs.to(self.device)
        actions = actions.to(self.device)
        next_obs = next_obs.to(self.device)
        theta_norm = theta_norm.to(self.device)

        end = self.ptr + batch_size
        if end <= self.capacity:
            self.obs[self.ptr:end].copy_(obs)
            self.actions[self.ptr:end].copy_(actions)
            self.next_obs[self.ptr:end].copy_(next_obs)
            self.theta_norm[self.ptr:end].copy_(theta_norm)
        else:
            first = self.capacity - self.ptr
            second = batch_size - first

            self.obs[self.ptr:self.capacity].copy_(obs[:first])
            self.actions[self.ptr:self.capacity].copy_(actions[:first])
            self.next_obs[self.ptr:self.capacity].copy_(next_obs[:first])
            self.theta_norm[self.ptr:self.capacity].copy_(theta_norm[:first])

            self.obs[0:second].copy_(obs[first:])
            self.actions[0:second].copy_(actions[first:])
            self.next_obs[0:second].copy_(next_obs[first:])
            self.theta_norm[0:second].copy_(theta_norm[first:])

        self.ptr = end % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def sample_batch(self, batch_size):
        if self.size == 0:
            raise ValueError("Cannot sample from empty replay buffer")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "next_obs": self.next_obs[indices],
            "theta_norm": self.theta_norm[indices],
        }
