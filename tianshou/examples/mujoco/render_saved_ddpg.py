import gym
import torch
import argparse
import numpy as np

import tianshou as ts
from tianshou.data import Collector
from tianshou.policy import DDPGPolicy
from tianshou.utils.net.common import Net
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.continuous import Actor, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="InvertedPendulum-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--start-timesteps", type=int, default=25000)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--render-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser.parse_args()


args = get_args()
env = gym.make(args.task, render_mode='human')
args.state_shape = env.observation_space.shape or env.observation_space.n
args.action_shape = env.action_space.shape or env.action_space.n
args.max_action = env.action_space.high[0]
args.exploration_noise = args.exploration_noise * args.max_action
print("Observations shape:", args.state_shape)
print("Actions shape:", args.action_shape)
print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# model
net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
actor = Actor(
    net_a, args.action_shape, max_action=args.max_action, device=args.device
).to(args.device)
actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
net_c = Net(
    args.state_shape,
    args.action_shape,
    hidden_sizes=args.hidden_sizes,
    concat=True,
    device=args.device,
)
critic = Critic(net_c, device=args.device).to(args.device)
critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
policy = DDPGPolicy(
    actor,
    actor_optim,
    critic,
    critic_optim,
    tau=args.tau,
    gamma=args.gamma,
    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    estimation_step=args.n_step,
    action_space=env.action_space,
)

model = policy.load_state_dict(torch.load(args.render_path, map_location=args.device))
policy.eval()
collector = Collector(policy, env)
collector.collect(n_episode=1, render=1/35)
