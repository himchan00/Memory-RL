from main import FLAGS
from policies.seq_models.mate_vanilla import Mate
from policies.seq_models.rnn_vanilla import LSTM
from policies.seq_models.gpt2_vanilla import GPT2

import torch
import argparse
import time
import torchkit.pytorch_utils as ptu
from torchkit.pytorch_utils import set_gpu_mode

device = 0
set_gpu_mode(torch.cuda.is_available(), device)
DEVICE = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

def instantiate_seq_model(args):
    if args.model == "mate":
        return Mate(args.hidden_size, args.hidden_size, args.n_layer, args.max_seq_length, 0.1).to(DEVICE)
    elif args.model == "mate_compile":
        return torch.compile(Mate(args.hidden_size, args.hidden_size, args.n_layer, args.max_seq_length, 0.1).to(DEVICE))
    elif args.model == "lstm":
        return LSTM(args.hidden_size, args.hidden_size, args.n_layer).to(DEVICE)
    elif args.model == "lstm_compile":
        return torch.compile(LSTM(args.hidden_size, args.hidden_size, args.n_layer).to(DEVICE))
    elif args.model == "gpt":
        return GPT2(args.hidden_size, args.hidden_size, args.n_layer, 1, 0.1, args.max_seq_length, "sine").to(DEVICE)
    elif args.model == "gpt_compile":
        return torch.compile(GPT2(args.hidden_size, args.hidden_size, args.n_layer, 1, 0.1, args.max_seq_length, "sine").to(DEVICE))
    else:
        raise ValueError(f"Unknown model: {args.model}")

def generate_random_input(args, seq_model):
    toy_input = ptu.randn(args.max_seq_length, args.batch_size, args.hidden_size)
    toy_h_0 = seq_model.get_zero_internal_state(
                batch_size=args.batch_size, training = True
            )
    return toy_input, toy_h_0

def check_inference_time(args, seq_model, toy_input, toy_h_0):
    seq_model.eval() # Set the model to evaluation mode (affects Dropout, BatchNorm, etc.)
    
    with torch.no_grad(): # Disable computation graph creation
        # 1. Warm-up (The first few operations may take longer due to memory allocation, kernel loading, JIT compilation, etc.)
        for _ in range(10):
            seq_model(toy_input, toy_h_0)
            
        # 2. Benchmark & Memory Tracking
        if torch.cuda.is_available():
            torch.cuda.synchronize() # Wait for GPU operations to complete. PyTorch CUDA operations are asynchronous, so synchronization is necessary for accurate time measurement.
            torch.cuda.reset_peak_memory_stats() # Reset the peak memory stats starting point
            
        start_time = time.perf_counter()

        for _ in range(100):
            seq_model(toy_input, toy_h_0)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize() # Wait for GPU operations to complete.
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) # Convert to MB
        else:
            peak_memory_mb = 0.0
            
        end_time = time.perf_counter()
        
    with open("./inference_time.txt", "a", encoding="utf-8") as f:
        f.write(f"Model: {args.model} | Hidden: {args.hidden_size} | Layer: {args.n_layer} | Seq: {args.max_seq_length} | Batch: {args.batch_size} | Params: {sum(p.numel() for p in seq_model.parameters())} | Inference time: {end_time - start_time}s | Peak VRAM: {peak_memory_mb} MB\n")

def check_backprop_time(args, seq_model, toy_input, toy_h_0):
    seq_model.train() # Set the model to training mode
    
    # 1. Warm-up (The first few operations may take longer due to memory allocation, kernel loading, JIT compilation, etc.)
    for _ in range(10):
        seq_model.zero_grad()
        out, _ = seq_model(toy_input, toy_h_0)
        random_output = ptu.randn(args.max_seq_length, args.batch_size, args.hidden_size)
        loss = torch.nn.functional.mse_loss(out, random_output)
        loss.backward()

    # 2. Benchmark
    back_time = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats() # Reset to track overall peak memory during Forward + Backward

    for _ in range(100):
        seq_model.zero_grad() # Prevent gradient accumulation (reset at each loop)
        
        out, _ = seq_model(toy_input, toy_h_0)
        random_output = ptu.randn(args.max_seq_length, args.batch_size, args.hidden_size)
        loss = torch.nn.functional.mse_loss(out, random_output)
        
        # Synchronize here if you want to measure purely the backward time
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        loss.backward()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()    
            
        end_time = time.perf_counter()

        back_time += (end_time - start_time)

    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_memory_mb = 0.0
        
    with open("./backprop_time.txt", "a", encoding="utf-8") as f:
        f.write(f"Model: {args.model} | Hidden: {args.hidden_size} | Layer: {args.n_layer} | Seq: {args.max_seq_length} | Batch: {args.batch_size} | Params: {sum(p.numel() for p in seq_model.parameters())} | Backprop time: {back_time}s | Peak VRAM: {peak_memory_mb} MB\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mate")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    seq_model = instantiate_seq_model(args)
    toy_input, toy_h_0 = generate_random_input(args, seq_model)
    
    check_inference_time(args, seq_model, toy_input, toy_h_0)
    check_backprop_time(args, seq_model, toy_input, toy_h_0)