from policies.seq_models.mate_vanilla import Mate
from policies.seq_models.rnn_vanilla import LSTM
from policies.seq_models.gpt2_vanilla import GPT2

import numpy as np
import torch
import argparse
import gc
import time
import torchkit.pytorch_utils as ptu
from torchkit.pytorch_utils import set_gpu_mode
torch.set_float32_matmul_precision('high') # Use TF32 for faster matmul

device = 0
set_gpu_mode(torch.cuda.is_available(), device)
DEVICE = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

def instantiate_seq_model(args):
    if args.model == "mate":
        return torch.compile(Mate(args.hidden_size, args.hidden_size, args.n_layer, args.max_seq_length, 0.1).to(DEVICE))
    elif args.model == "lstm":
        return torch.compile(LSTM(args.hidden_size, args.hidden_size, args.n_layer).to(DEVICE))
    elif args.model == "gpt":
        return torch.compile(GPT2(args.hidden_size, args.hidden_size, args.n_layer, 1, 0.1, args.max_seq_length, "sine").to(DEVICE))
    else:
        raise ValueError(f"Unknown model: {args.model}")


def check_rollout_time(args, seq_model):
    seq_model.eval()
    toy_input = ptu.randn(1, args.batch_size, args.hidden_size).to(DEVICE)
    
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
                    f"Seq: {args.max_seq_length} | Batch: {args.batch_size} | "
                    f"Params: {sum(p.numel() for p in seq_model.parameters())} | "
                    f"Rollout time: {rollout_time_mean:.4f} +- {rollout_time_std:.4f} s | "
                    f"Peak VRAM: {peak_memory_mean:.2f} +- {peak_memory_std:.2f} MB\n"
                )

def check_update_time(args, seq_model):
    seq_model.train() # Set the model to training mode
    toy_input = ptu.randn(args.max_seq_length, args.batch_size, args.hidden_size)
    toy_internal_state = seq_model.get_zero_internal_state(
                batch_size=args.batch_size, training = True
            )

    # 1. Warm-up (The first few operations may take longer due to memory allocation, kernel loading, JIT compilation, etc.)
    for _ in range(3):
        seq_model.zero_grad()
        out, _ = seq_model(toy_input, toy_internal_state)
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
                f"Seq: {args.max_seq_length} | Batch: {args.batch_size} | "
                f"Params: {sum(p.numel() for p in seq_model.parameters())} | "
                f"Update time: {update_time_mean:.4f} +- {update_time_std:.4f} s | "
                f"Peak VRAM: {peak_memory_mean:.2f} +- {peak_memory_std:.2f} MB\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mate")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mode", type=str, default="rollout", choices=["rollout", "update"])
    args = parser.parse_args()
    
    seq_model = instantiate_seq_model(args)
    if args.mode == "rollout":
        check_rollout_time(args, seq_model)
    elif args.mode == "update":
        check_update_time(args, seq_model)