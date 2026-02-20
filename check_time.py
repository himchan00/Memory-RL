from policies.seq_models.mate_vanilla import Mate
from policies.seq_models.rnn_vanilla import LSTM

import torch
import argparse
import time
import torchkit.pytorch_utils as ptu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def instantiate_seq_model(args):
    if args.model == "mate":
        return Mate(args.hidden_size, args.hidden_size, args.n_layer, args.max_seq_length, 0.1).to(DEVICE)
        #return torch.compile(Mate(args.hidden_size, args.hidden_size, args.n_layer, args.max_seq_length, 0.1).to(DEVICE))
    elif args.model == "lstm":
        return LSTM(args.hidden_size, args.hidden_size, args.n_layer).to(DEVICE)
        #return torch.compile(LSTM(args.hidden_size, args.hidden_size, args.n_layer).to(DEVICE))
    else:
        raise ValueError(f"Unknown model: {args.model}")



def generate_random_input(args):
    toy_input = ptu.randn(args.max_seq_length, args.batch_size, args.hidden_size).to(DEVICE)
    toy_h_0 = ptu.randn(1, args.batch_size, args.hidden_size).to(DEVICE)
    if args.model == "lstm":
        toy_c_0 =ptu.randn(1, args.batch_size, args.hidden_size).to(DEVICE)
        toy_h_0 = (toy_h_0, toy_c_0)
    return toy_input, toy_h_0

def check_inference_time(args,seq_model, toy_input, toy_h_0):
    start_time = time.perf_counter()
    for _ in range(100):
        seq_model(toy_input, toy_h_0)
    end_time = time.perf_counter()
    with open("./forward_time.txt", "a", encoding="utf-8") as f:
        f.write(f"Model: {args.model} {args.hidden_size} {args.n_layer} {args.max_seq_length} {args.batch_size} Number of parameters: {sum(p.numel() for p in seq_model.parameters())}       Inference time: {end_time - start_time}\n")

def check_backprop_time(args,seq_model, toy_input, toy_h_0):
    back_time = 0.0
    for _ in range(100):
        out, _ = seq_model(toy_input, toy_h_0)
        random_output = ptu.randn(args.max_seq_length, args.batch_size, args.hidden_size).to(DEVICE)
        loss = torch.nn.functional.mse_loss(out, random_output)
        start_time = time.perf_counter()
        loss.backward()
        end_time = time.perf_counter()
        back_time+=(end_time - start_time)
    with open("./backprop_time.txt", "a", encoding="utf-8") as f:
        f.write(f"Model: {args.model} {args.hidden_size} {args.n_layer} {args.max_seq_length} {args.batch_size} Number of parameters: {sum(p.numel() for p in seq_model.parameters())}      Backprop time: {back_time}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mate")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    seq_model = instantiate_seq_model(args)
    toy_input, toy_h_0 = generate_random_input(args)
    check_inference_time(args,seq_model, toy_input, toy_h_0)
    check_backprop_time(args,seq_model, toy_input, toy_h_0)
