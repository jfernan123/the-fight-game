"""Train the walker with PPO on GPU-batched MuJoCo Warp environments (walk_env_gpu.py).

stable-baselines3 doesn't speak batched-tensor GPU envs, so this is a from-scratch PyTorch PPO
implementation (CleanRL-style: single actor-critic MLP, GAE advantages, clipped surrogate loss)
running against thousands of environments stepped together each iteration.

    python train_gpu.py --timesteps 20000000
    python train_gpu.py --ankle dual --num-envs 2048

Progress: tensorboard --logdir tb_logs_gpu

Checkpoints/logs default to runs_gpu/<mode>/<run-id>/ and tb_logs_gpu/<mode>/<run-id>/ (mode =
"baseline" if --ankle is omitted, run-id = a timestamp) -- every launch gets its own folder, so
successive runs never collide. Confirmed this actually happens if they don't: relaunching the same
--ankle without a fresh run-id wrote a second TensorBoard event file into the same directory as a
prior run, and TensorBoard's default behavior on seeing step numbers jump backward (old run's late
steps, then new run restarting at 0) produces a genuinely garbled merged graph, not just visual
clutter. watch.py/watch_gpu.py/watch_all.py search recursively so they still find the newest
checkpoint regardless of which run-id folder it's in. Pass --checkpoint-dir/--tensorboard-log
explicitly to override (e.g. to intentionally continue writing into one fixed folder).

To train all four ankle variants, use train_gpu_all.py -- it runs this script once per variant,
SEQUENTIALLY (not in parallel like train_all.py's CPU runs, since all four would otherwise fight
over the same one GPU).
"""

from __future__ import annotations

import argparse
import os
import re
import time

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from walk_env_gpu import WalkEnvGPU


# Bounds for actor_log_std -- unlike a discrete policy (entropy capped at log(num_actions)), a
# continuous Gaussian policy's entropy is unbounded above, so an entropy bonus with no ceiling on
# std can make the optimizer just inflate log_std forever instead of solving the task, since that's
# a cheaper way to raise entropy than actually improving the policy. Confirmed empirically: with
# --entropy-coef 0.05 and no clamp, entropy climbed monotonically from ~10 to ~39 nats over 66M
# steps (implying std ~630 per action dim, i.e. near-pure noise clipped into [-1,1]), while
# clip_fraction collapsed to ~0 and ep_rew_mean never trended up -- the policy wasn't converging at
# any step count, it was diverging. [-20, 2] is the standard range used for this everywhere
# (SpinningUp's SAC, CleanRL's continuous PPO variants).
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        mean = self.actor_mean(obs)
        log_std = self.actor_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the walker with PPO on GPU-batched MuJoCo Warp envs.")
    parser.add_argument("--timesteps", type=int, default=20_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--unroll-length", type=int, default=32, help="steps collected per env before each PPO update")
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ankle", choices=["single", "dual", "detailed"], default=None)
    parser.add_argument("--checkpoint-dir", default=None, help="default: runs_gpu/<mode>/<run-id>/ (run-id = timestamp)")
    parser.add_argument("--tensorboard-log", default=None, help="default: tb_logs_gpu/<mode>/<run-id>/ (run-id = timestamp)")
    parser.add_argument("--save-freq", type=int, default=2_000_000, help="save a checkpoint every N env-steps")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resume", default=None,
        help="Path to a .pt checkpoint to warm-start the network from (weights only, optimizer "
             "starts fresh). --timesteps is how many MORE steps this run does, not a new total. "
             "If the filename is walker_<N>_steps.pt, N is used as the starting point for "
             "TensorBoard's step axis so the curve continues instead of restarting at 0 -- "
             "walker_final.pt has no encoded count, use --start-step to set it explicitly.",
    )
    parser.add_argument("--start-step", type=int, default=None, help="override the TensorBoard step offset when resuming")
    args = parser.parse_args()
    mode = args.ankle or "baseline"
    run_id = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = args.checkpoint_dir or os.path.join("runs_gpu", mode, run_id)
    tensorboard_log = args.tensorboard_log or os.path.join("tb_logs_gpu", mode, run_id)

    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_log)

    env = WalkEnvGPU(ankle_mode=args.ankle, num_envs=args.num_envs, device=args.device)
    agent = ActorCritic(env.obs_dim, env.action_dim).to(args.device)

    step_offset = 0
    if args.resume:
        agent.load_state_dict(torch.load(args.resume, map_location=args.device))
        print(f"Resumed weights from {args.resume}")
        if args.start_step is not None:
            step_offset = args.start_step
        else:
            match = re.search(r"walker_(\d+)_steps\.pt$", args.resume)
            step_offset = int(match.group(1)) if match else 0
            if match:
                print(f"Continuing TensorBoard step axis from {step_offset:,} (parsed from filename)")
            else:
                print("Could not parse a step count from --resume's filename -- TensorBoard step axis "
                      "starts at 0 for this run (pass --start-step to set it explicitly).")

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    batch_size = args.num_envs * args.unroll_length
    minibatch_size = batch_size // args.num_minibatches

    obs_buf = torch.zeros(args.unroll_length, args.num_envs, env.obs_dim, device=args.device)
    action_buf = torch.zeros(args.unroll_length, args.num_envs, env.action_dim, device=args.device)
    logprob_buf = torch.zeros(args.unroll_length, args.num_envs, device=args.device)
    reward_buf = torch.zeros(args.unroll_length, args.num_envs, device=args.device)
    done_buf = torch.zeros(args.unroll_length, args.num_envs, device=args.device)
    value_buf = torch.zeros(args.unroll_length, args.num_envs, device=args.device)

    obs = env.reset()
    done = torch.zeros(args.num_envs, device=args.device)

    ep_return = torch.zeros(args.num_envs, device=args.device)
    ep_len = torch.zeros(args.num_envs, device=args.device)
    recent_returns: list[float] = []
    recent_lens: list[float] = []

    global_step = step_offset
    start_time = time.time()
    num_updates = max(1, args.timesteps // batch_size)
    next_save = step_offset + args.save_freq

    # Running mean of each reward term over the rollout, logged to TensorBoard as reward/<term>.
    # Worth the few lines: two separate bugs in this project (a foot-slip term quietly dominating
    # at -0.42/step, and velocity tracking flatlined at 0.0001 with no usable gradient) were both
    # invisible in ep_rew_mean and immediately obvious once the terms were plotted separately.
    term_sums: dict[str, float] = {}
    term_count = 0

    for update in range(1, num_updates + 1):
        for t in range(args.unroll_length):
            global_step += args.num_envs
            obs_buf[t] = obs
            done_buf[t] = done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)
            action_buf[t] = action
            logprob_buf[t] = logprob
            value_buf[t] = value

            obs, reward, fallen, truncated = env.step(action)
            done = (fallen | truncated).float()
            reward_buf[t] = reward

            for term_name, term_value in env.reward_breakdown.items():
                term_sums[term_name] = term_sums.get(term_name, 0.0) + float(term_value.mean())
            term_count += 1

            ep_return += reward
            ep_len += 1
            done_mask = done.bool()
            if done_mask.any():
                recent_returns.extend(ep_return[done_mask].tolist())
                recent_lens.extend(ep_len[done_mask].tolist())
                ep_return[done_mask] = 0.0
                ep_len[done_mask] = 0.0

        with torch.no_grad():
            next_value = agent.get_value(obs)
            advantages = torch.zeros_like(reward_buf)
            last_gae = torch.zeros(args.num_envs, device=args.device)
            for t in reversed(range(args.unroll_length)):
                if t + 1 < args.unroll_length:
                    next_non_terminal = 1.0 - done_buf[t + 1]
                    next_val = value_buf[t + 1]
                else:
                    next_non_terminal = 1.0 - done
                    next_val = next_value
                delta = reward_buf[t] + args.gamma * next_val * next_non_terminal - value_buf[t]
                last_gae = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae
            returns = advantages + value_buf

        b_obs = obs_buf.reshape(-1, env.obs_dim)
        b_actions = action_buf.reshape(-1, env.action_dim)
        b_logprobs = logprob_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = value_buf.reshape(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        clipfracs = []
        for _epoch in range(args.update_epochs):
            perm = torch.randperm(batch_size, device=args.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]
                _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs[mb_idx], b_actions[mb_idx])
                log_ratio = new_logprob - b_logprobs[mb_idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_idx]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.entropy_coef * entropy_loss + args.value_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        with torch.no_grad():
            explained_var = 1 - (b_returns - b_values).var() / (b_returns.var() + 1e-8)

        elapsed = time.time() - start_time
        fps = int(global_step / elapsed)
        writer.add_scalar("train/value_loss", value_loss.item(), global_step)
        writer.add_scalar("train/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("train/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("train/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("train/clip_fraction", sum(clipfracs) / len(clipfracs), global_step)
        writer.add_scalar("train/explained_variance", explained_var.item(), global_step)
        writer.add_scalar("time/fps", fps, global_step)
        for term_name, term_total in term_sums.items():
            writer.add_scalar(f"reward/{term_name}", term_total / max(term_count, 1), global_step)
        term_sums.clear()
        term_count = 0
        if recent_returns:
            ep_rew_mean = sum(recent_returns) / len(recent_returns)
            ep_len_mean = sum(recent_lens) / len(recent_lens)
            writer.add_scalar("rollout/ep_rew_mean", ep_rew_mean, global_step)
            writer.add_scalar("rollout/ep_len_mean", ep_len_mean, global_step)
            print(
                f"step {global_step:>12,d}  ep_rew_mean={ep_rew_mean:8.2f}  ep_len_mean={ep_len_mean:7.1f}  "
                f"fps={fps:,}"
            )
            recent_returns.clear()
            recent_lens.clear()
        writer.flush()

        if global_step >= next_save:
            path = os.path.join(checkpoint_dir, f"walker_{global_step}_steps.pt")
            torch.save(agent.state_dict(), path)
            next_save += args.save_freq

    final_path = os.path.join(checkpoint_dir, "walker_final.pt")
    torch.save(agent.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
