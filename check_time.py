from policies.seq_models.rnn_vanilla import LSTM
from policies.seq_models.gpt2_vanilla import GPT2
from policies.seq_models.splagger import SplAgger
from policies.seq_models.mamba_vanilla import Mamba
from policies.seq_models.mate_vanilla import build_mate_embedder

import numpy as np
import torch
import torch.nn as nn
import argparse
import gc
import time
import torchkit.pytorch_utils as ptu
from torchkit.pytorch_utils import set_gpu_mode
torch.set_float32_matmul_precision('high') # Use TF32 for faster matmul

device = 0
set_gpu_mode(torch.cuda.is_available(), device)
DEVICE = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Defaults mirrored from the repo configs:
#   configs/seq_models/common.py           -> dropout_emb / dropout_ff, full_transition
#   configs/seq_models/mate_default.py     -> n_layer, hidden_size, learn_init_emb
#   configs/seq_models/gpt_default.py      -> n_layer, n_head, position_encoding
#   configs/seq_models/lstm_default.py     -> n_layer, hidden_size
#   configs/seq_models/splagger_default.py -> n_layer, hidden_size, agg_type
#   configs/seq_models/mamba_default.py    -> n_layer, hidden_size, d_state/d_conv/expand
#
# SCOPE: the measured module is `transition_embedder + seq_model`, wired the way
# `RNN_head` wires it. The projection is part of what actually runs and its cost
# is consumer-dependent -- Inductor folds the activation into GPT2's own Triton
# kernels but cannot fold it into cuDNN's opaque LSTM/GRU cell -- so leaving it
# out would not be neutral across models. `InputNorm`, the obs embedder and the
# RL heads are excluded (identical across models, constant offset).
# ---------------------------------------------------------------------------
DROPOUT_EMB = 0.05      # config_seq.dropout_emb
DROPOUT_FF = 0.05       # config_seq.dropout_ff
FULL_TRANSITION = True  # config_seq.full_transition
N_HEAD = 1              # config_seq.seq_model.n_head (gpt)
POSITION_ENCODING = "sine"  # config_seq.seq_model.position_encoding (gpt)


class MateBench(nn.Module):
    """MATE reduced to the ops a timing run needs: embedder + init prior + running mean.

    Matches `Mate` on the `learn_init_emb=True` path (the config default) and
    drops everything that costs kernel launches without being part of the memory
    mechanism: the `info` dict (`init_emb.norm()` and `log_init_weight.exp()` are
    two extra launches per rollout step that `RNN_head.step` discards), MSC,
    `z_norm`, the EMA embedder and the rollout z-cache -- none of which a plain
    `mate_default.py` run enables.

    Kept deliberately independent of `Mate` so the benchmark does not drift with
    experimental knobs; re-check it against `Mate.forward` if the core
    cumsum/count math there changes.
    """

    name = "mate"

    def __init__(self, input_size, hidden_size, n_layer, dropout_emb, dropout_ff, embedder_type="mlp"):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedder = build_mate_embedder(
            embedder_type, input_size, hidden_size, n_layer, dropout_emb, dropout_ff,
        )
        # Initial-memory prior: m_t = (w * init_emb + sum_i z_i) / (w + t)
        self.init_emb = nn.Parameter(ptu.randn(hidden_size))
        self.log_init_weight = nn.Parameter(ptu.zeros(()))

    def forward(self, inputs, h_0, **kwargs):
        hidden, initial_count = h_0
        z = self.embedder(inputs)
        # cat([init, z]).cumsum(0)[1:] rather than init + z.cumsum(0): mirrors
        # Mate, avoids an Inductor SplitScan crash (pytorch/pytorch#180221)
        cumsum = torch.cat([hidden, z], dim=0).cumsum(dim=0)[1:]
        step_counts = torch.arange(
            1, z.shape[0] + 1, device=initial_count.device, dtype=initial_count.dtype,
        ).view(-1, 1, 1)
        counts = initial_count + step_counts
        output = cumsum / counts.clamp(min=1e-6)
        return output, (cumsum[-1].clone().unsqueeze(0), counts[-1].clone().unsqueeze(0))

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        t_0 = self.log_init_weight.exp().view(1, 1, 1).expand(1, batch_size, 1)
        return self.init_emb.view(1, 1, -1).expand(1, batch_size, -1) * t_0, t_0


class BenchmarkModel(nn.Module):
    """`RNN_head`'s transition embedder + sequence model; normalizes the return arity."""

    def __init__(self, transition_embedder, seq_model):
        super().__init__()
        self.transition_embedder = transition_embedder
        self.seq_model = seq_model

    def forward(self, transitions, internal_state, **kwargs):
        # Sequence models return either (output, h_n) or (output, h_n, info).
        ret = self.seq_model(self.transition_embedder(transitions), internal_state, **kwargs)
        return ret[0], ret[1]

    def get_zero_internal_state(self, **kwargs):
        return self.seq_model.get_zero_internal_state(**kwargs)


def transition_size(args):
    """`RNN_head.__init__`: 2 * obs_dim + action_dim + 1 when full_transition."""
    if FULL_TRANSITION:
        return 2 * args.obs_dim + args.act_dim + 1
    return args.obs_dim + args.act_dim + 1


def instantiate_seq_model(args):
    """Build `transition_embedder + seq_model` on DEVICE, at repo default settings."""
    h, n = args.hidden_size, args.n_layer
    in_size = transition_size(args)
    # mate/gpt set seq_model.max_seq_length = max_episode_steps + 1
    # (mate_update_fn / gpt_update_fn), so the model sees one slot per real
    # transition plus the t=-1 alignment step.
    model_max_seq_length = args.max_seq_length + 1

    if args.model == "mate":
        # RNN_head gives mate an IdentityModule: MATE owns its input projection.
        embedder = nn.Identity()
        seq_model = MateBench(in_size, h, n, DROPOUT_EMB, DROPOUT_FF, args.embedder_type)
    else:
        embedder = nn.Sequential(
            nn.Linear(in_size, h), nn.LeakyReLU(), nn.Dropout(DROPOUT_EMB)
        )
        if args.model == "lstm":
            seq_model = LSTM(h, h, n, dropout_ff=DROPOUT_FF)
        elif args.model == "gpt":
            seq_model = GPT2(
                h, h, n, N_HEAD, DROPOUT_EMB, DROPOUT_FF,
                model_max_seq_length, POSITION_ENCODING,
            )
        elif args.model == "splagger":
            seq_model = SplAgger(h, h, n, dropout_ff=DROPOUT_FF, agg_type=args.agg_type)
        elif args.model == "mamba":
            seq_model = Mamba(h, h, n, dropout_ff=DROPOUT_FF)
        else:
            raise ValueError(f"Unknown model: {args.model}")

    return torch.compile(BenchmarkModel(embedder, seq_model).to(DEVICE))


def check_rollout_time(args, seq_model):
    seq_model.eval()
    toy_input = ptu.randn(1, args.batch_size, transition_size(args)).to(DEVICE)
    
    with torch.no_grad():
        # 1. Warm-up (The first few operations may take longer due to memory allocation, kernel loading, JIT compilation, etc.)
        for _ in range(3):
            toy_internal_state = seq_model.get_zero_internal_state(batch_size=args.batch_size)
            for _ in range(args.max_seq_length):
                _, toy_internal_state = seq_model(toy_input, toy_internal_state)

        # 2. Benchmark
        rollout_time = []
        peak_memory = []
        n_trials = 10
    
        gc.disable()
        for _ in range(n_trials):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats() 

            toy_internal_state = seq_model.get_zero_internal_state(batch_size=args.batch_size)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            
            for _ in range(args.max_seq_length):
                _, toy_internal_state = seq_model(toy_input, toy_internal_state)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            rollout_time.append(end_time - start_time)

            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                peak_memory_mb = 0.0
            peak_memory.append(peak_memory_mb)
        gc.enable()
        # Calculate Mean and Std
        rollout_time_mean = np.mean(rollout_time)
        rollout_time_std = np.std(rollout_time)
        peak_memory_mean = np.mean(peak_memory)
        peak_memory_std = np.std(peak_memory)

        # Write to file with "+-" format
        with open("./rollout_time.txt", "a", encoding="utf-8") as f:
            f.write(f"Model: {args.model} | Hidden: {args.hidden_size} | Layer: {args.n_layer} | "
                    f"Trans: {transition_size(args)} | "
                    f"Seq: {args.max_seq_length} | Batch: {args.batch_size} | "
                    f"Params: {sum(p.numel() for p in seq_model.parameters())} | "
                    f"Rollout time: {rollout_time_mean:.4f} +- {rollout_time_std:.4f} s | "
                    f"Peak VRAM: {peak_memory_mean:.2f} +- {peak_memory_std:.2f} MB\n"
                )

def check_update_time(args, seq_model):
    seq_model.train() # Set the model to training mode
    toy_input = ptu.randn(args.max_seq_length, args.batch_size, transition_size(args))

    # Rebuilt every iteration, like RNN_head.forward does: the initial state is
    # part of the autograd graph (get_zero_internal_state reads init_emb /
    # log_init_weight), so reusing one state across backward calls would try to
    # backward through the same graph twice.
    def zero_state():
        return seq_model.get_zero_internal_state(
            batch_size=args.batch_size, training=True
        )

    # 1. Warm-up (The first few operations may take longer due to memory allocation, kernel loading, JIT compilation, etc.)
    for _ in range(3):
        seq_model.zero_grad()
        out, _ = seq_model(toy_input, zero_state())
        loss = (out ** 2).sum(dim=-1).mean() # Dummy loss
        loss.backward()

    # 2. Benchmark
    update_time = []
    peak_memory = []
    n_trials = 10

    gc.disable()
    for _ in range(n_trials):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats() # Reset to track overall peak memory during Forward + Backward
        seq_model.zero_grad() # Prevent gradient accumulation (reset at each loop)
        toy_internal_state = zero_state()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        out, _ = seq_model(toy_input, toy_internal_state)
        loss = (out ** 2).sum(dim=-1).mean() # Dummy loss
        loss.backward()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()    
            
        end_time = time.perf_counter()
        update_time.append(end_time - start_time)

        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_memory_mb = 0.0
        peak_memory.append(peak_memory_mb)
    gc.enable()
    # Calculate Mean and Std
    update_time_mean = np.mean(update_time)
    update_time_std = np.std(update_time)
    peak_memory_mean = np.mean(peak_memory)
    peak_memory_std = np.std(peak_memory)
        
    with open("./update_time.txt", "a", encoding="utf-8") as f:
        f.write(f"Model: {args.model} | Hidden: {args.hidden_size} | Layer: {args.n_layer} | "
                f"Trans: {transition_size(args)} | "
                f"Seq: {args.max_seq_length} | Batch: {args.batch_size} | "
                f"Params: {sum(p.numel() for p in seq_model.parameters())} | "
                f"Update time: {update_time_mean:.4f} +- {update_time_std:.4f} s | "
                f"Peak VRAM: {peak_memory_mean:.2f} +- {peak_memory_std:.2f} MB\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mate",
                        choices=["mate", "lstm", "gpt", "splagger", "mamba"])
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    # T-Maze passive defaults (obs_dim=3, Discrete(4) -> one-hot act_dim=4);
    # transition_size = 2 * obs_dim + act_dim + 1, as in RNN_head.
    parser.add_argument("--obs_dim", type=int, default=3)
    parser.add_argument("--act_dim", type=int, default=4)
    parser.add_argument("--agg_type", type=str, default="max", choices=["max", "mean"],
                        help="splagger causal aggregator")
    parser.add_argument("--embedder_type", type=str, default="mlp", choices=["mlp", "gpt_ffn"],
                        help="mate embedder")
    parser.add_argument("--mode", type=str, default="rollout", choices=["rollout", "update"])
    args = parser.parse_args()
    
    seq_model = instantiate_seq_model(args)
    if args.mode == "rollout":
        check_rollout_time(args, seq_model)
    elif args.mode == "update":
        check_update_time(args, seq_model)