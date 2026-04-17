"""Train PufferLib's default Craftax-Classic policy on craftax.c.

Observation layout (1345 floats = 7x9x21 map + 22 flat) matches the JAX env
exactly, so `pufferlib.environments.craftax.torch.Policy` runs unchanged.

Environment overrides:
    CRAFTAX_NUM_ENVS      (default 1024)   envs per worker
    CRAFTAX_TIMESTEPS     (default 1_000_000)
    CRAFTAX_BATCH         (default 65536)
    CRAFTAX_MINIBATCH     (default 4096)

Usage:
    make libcraftax.so
    CRAFTAX_LIB=$PWD/libcraftax.so PYTHONPATH=python \\
        /path/to/pufferlib/venv/bin/python python/train_craftax_c.py
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")

import torch
import pufferlib
import pufferlib.vector
from pufferlib import pufferl
from pufferlib.environments.craftax.torch import Policy as _Policy

from craftax_c.environment import env_creator as craftax_c_env_creator


class Policy(_Policy):
    # Puffer's non-RNN train loop calls `policy(obs, state)` with state=None.
    def forward(self, observations, state=None):
        hidden, lookup = self.encode_observations(observations)
        action, value = self.decode_actions(hidden, lookup)
        return action, value

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)


NUM_ENVS  = int(os.environ.get("CRAFTAX_NUM_ENVS", 1024))
TIMESTEPS = int(os.environ.get("CRAFTAX_TIMESTEPS", 1_000_000))
BATCH     = int(os.environ.get("CRAFTAX_BATCH", 65536))
MINIBATCH = int(os.environ.get("CRAFTAX_MINIBATCH", 4096))
DEVICE    = os.environ.get("CRAFTAX_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


env_name = "Craftax-C-Symbolic-v1"
make_env = craftax_c_env_creator(env_name)

vecenv = pufferlib.vector.make(
    make_env,
    num_envs=1,
    num_workers=1,
    batch_size=1,
    backend=pufferlib.vector.Serial,
    env_kwargs={"num_envs": NUM_ENVS},
)

policy = Policy(vecenv.driver_env).to(DEVICE)

# Reuse PufferLib's Craftax training hyperparams (PPO, GAE, etc.).
args = pufferl.load_config("Craftax-Classic-Symbolic-v1")
args["train"]["env"] = env_name
args["train"]["device"] = DEVICE
args["train"]["total_timesteps"] = TIMESTEPS
args["train"]["batch_size"] = BATCH
args["train"]["minibatch_size"] = MINIBATCH
args["train"]["use_rnn"] = False

trainer = pufferl.PuffeRL(args["train"], vecenv, policy)

while trainer.epoch < trainer.total_epochs:
    trainer.evaluate()
    if DEVICE == "cuda":
        torch.compiler.cudagraph_mark_step_begin()
    trainer.train()

trainer.print_dashboard()
trainer.close()
