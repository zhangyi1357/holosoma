#!/usr/bin/env python3
"""Microbenchmark eager versus torch.compile for the A3/G1 FastSAC network shapes."""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(154, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.mu = nn.Linear(128, 29)
        self.log_std = nn.Linear(128, 29)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(obs)
        mean = self.mu(hidden)
        log_std = self.log_std(hidden).clamp(-5.0, 0.0)
        action = torch.tanh(mean + log_std.exp() * torch.randn_like(mean))
        return action, log_std


class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qnets = nn.ModuleList([self._make_qnet() for _ in range(2)])

    @staticmethod
    def _make_qnet() -> nn.Module:
        return nn.Sequential(
            nn.Linear(315, 768),
            nn.LayerNorm(768),
            nn.SiLU(),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, 501),
        )

    def forward(self, critic_obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = torch.cat((critic_obs, action), dim=-1)
        return torch.stack([network(features) for network in self.qnets])


def _measure(function, iterations: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(1)
    device = torch.device("cuda")
    actor = Actor().to(device)
    critic = Critic().to(device)
    target = Critic().to(device).eval()
    target.load_state_dict(critic.state_dict())
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=3e-4, fused=True)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=3e-4, fused=True)
    obs = torch.randn(args.batch_size, 154, device=device)
    next_obs = torch.randn_like(obs)
    critic_obs = torch.randn(args.batch_size, 286, device=device)
    next_critic_obs = torch.randn_like(critic_obs)
    replay_action = torch.randn(args.batch_size, 29, device=device).tanh()

    def critic_step() -> torch.Tensor:
        critic_optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                next_action, _ = actor(next_obs)
                target_probability = F.softmax(target(next_critic_obs, next_action), dim=-1)
            prediction = critic(critic_obs, replay_action)
            loss = -(target_probability * F.log_softmax(prediction, dim=-1)).sum(dim=-1).mean()
        loss.backward()
        critic_optimizer.step()
        return loss

    def actor_step() -> torch.Tensor:
        actor_optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action, log_std = actor(obs)
            probability = F.softmax(critic(critic_obs, action), dim=-1)
            support = torch.linspace(-20.0, 20.0, 501, device=device)
            q_value = (probability * support).sum(dim=-1).mean(dim=0)
            loss = (0.001 * log_std.sum(dim=-1) - q_value).mean()
        loss.backward()
        actor_optimizer.step()
        return loss

    def update() -> None:
        critic_step()
        actor_step()

    for _ in range(args.warmup):
        update()
    eager_seconds = _measure(update, args.iterations)

    compiled_critic = torch.compile(critic_step)
    compiled_actor = torch.compile(actor_step)

    def compiled_update() -> None:
        compiled_critic()
        compiled_actor()

    compile_start = time.perf_counter()
    for _ in range(args.warmup):
        compiled_update()
    torch.cuda.synchronize()
    compile_warmup_seconds = time.perf_counter() - compile_start
    compiled_seconds = _measure(compiled_update, args.iterations)

    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} batch={args.batch_size}")
    print(f"eager_ms={eager_seconds * 1000.0:.3f}")
    print(f"compiled_ms={compiled_seconds * 1000.0:.3f}")
    print(f"steady_state_speedup={eager_seconds / compiled_seconds:.3f}x")
    print(f"compile_warmup_seconds={compile_warmup_seconds:.3f}")


if __name__ == "__main__":
    main()
