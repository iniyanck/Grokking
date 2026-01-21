import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import ModularAdditionDataset, PermutationDataset, MatrixDataset, DateDataset, TicTacToeDataset, GameOfLifeDataset
from model import SimpleTransformer
import argparse
import time
import json
import os

def train():
    # Parse config path first to load defaults
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default='config.json')
    pre_args, _ = pre_parser.parse_known_args()

    # Define default values
    defaults = {
        'p': 113,
        'epochs': 5000,
        'batch_size': 512,
        'lr': 1e-3,
        'weight_decay': 1.0,
        'd_model': 128,
        'n_layers': 1,
        'n_heads': 4,
        'seed': 42,
        'dataset': 'modular',
        'matrix_p': 4,
        'matrix_n': 2
    }

    # Load config file if it exists
    config_path = pre_args.config
    # Check current dir, then parent dir (if running from src)
    if not os.path.exists(config_path) and os.path.exists(os.path.join('..', config_path)):
        config_path = os.path.join('..', config_path)

    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            file_config = json.load(f)
            defaults.update(file_config)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--p', type=int, default=defaults['p'])
    parser.add_argument('--epochs', type=int, default=defaults['epochs']) # Grokking can take a while
    parser.add_argument('--batch_size', type=int, default=defaults['batch_size'])
    parser.add_argument('--lr', type=float, default=defaults['lr'])
    parser.add_argument('--weight_decay', type=float, default=defaults['weight_decay']) # High WD is key per Power et al. (2022)
    parser.add_argument('--d_model', type=int, default=defaults['d_model'])
    parser.add_argument('--n_layers', type=int, default=defaults['n_layers'])
    parser.add_argument('--n_heads', type=int, default=defaults['n_heads'])
    parser.add_argument('--seed', type=int, default=defaults['seed'])
    
    parser.add_argument('--dataset', type=str, default=defaults['dataset'], choices=['modular', 'permutation', 'matrix', 'date', 'tictactoe', 'gol'])
    parser.add_argument('--matrix_p', type=int, default=defaults['matrix_p'], help='Modulus for matrix elements')
    parser.add_argument('--matrix_n', type=int, default=defaults['matrix_n'], help='Dimension of square matrix')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    if args.dataset == 'modular':
        train_dataset = ModularAdditionDataset(p=args.p, split='train', train_fraction=0.5, seed=args.seed)
        val_dataset = ModularAdditionDataset(p=args.p, split='val', train_fraction=0.5, seed=args.seed)
        vocab_size = args.p
    elif args.dataset == 'permutation':
        # args.p is ignored for permutation, fixed to S_5 for now (size 120)
        train_dataset = PermutationDataset(n=5, split='train', train_fraction=0.5, seed=args.seed)
        val_dataset = PermutationDataset(n=5, split='val', train_fraction=0.5, seed=args.seed)
        vocab_size = train_dataset.cardinality # Should be 120
    elif args.dataset == 'matrix':
        train_dataset = MatrixDataset(p=args.matrix_p, n=args.matrix_n, split='train', train_fraction=0.5, seed=args.seed)
        val_dataset = MatrixDataset(p=args.matrix_p, n=args.matrix_n, split='val', train_fraction=0.5, seed=args.seed)
        vocab_size = train_dataset.cardinality
    elif args.dataset == 'date':
        train_dataset = DateDataset(split='train', train_fraction=0.5, seed=args.seed)
        val_dataset = DateDataset(split='val', train_fraction=0.5, seed=args.seed)
        vocab_size = 2200 # efficient enough covers years up to 2100
    elif args.dataset == 'tictactoe':
        train_dataset = TicTacToeDataset(split='train', train_fraction=0.5, seed=args.seed)
        val_dataset = TicTacToeDataset(split='val', train_fraction=0.5, seed=args.seed)
        vocab_size = 3 # 0, 1, 2
    elif args.dataset == 'gol':
        train_dataset = GameOfLifeDataset(split='train', train_fraction=0.5, seed=args.seed)
        val_dataset = GameOfLifeDataset(split='val', train_fraction=0.5, seed=args.seed)
        vocab_size = 2 # 0, 1

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False) # Full batch for val
    
    # Model
    
    # Determine max_len from dataset
    # Get one sample to check length
    sample_x, _ = train_dataset[0]
    max_len = sample_x.size(0)
    print(f"Dataset {args.dataset}: vocab_size={vocab_size}, max_len={max_len}")

    model = SimpleTransformer(vocab_size=vocab_size, d_model=args.d_model, 
                              n_head=args.n_heads, n_layers=args.n_layers, max_len=max_len)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Logging
    history = {'train_acc': [], 'val_acc': [], 'step': []}
    
    start_time = time.time()
    
    step = 0
    for epoch in range(args.epochs):
        model.train()
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            step += 1
            
        train_acc = 100 * correct / total
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct = (predicted == targets).sum().item()
                    val_total = targets.size(0)
                    val_acc = 100 * val_correct / val_total
            
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['step'].append(step)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Step {step}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%")
                
                # Check for grokking/convergence
                if val_acc > 99.5:
                    print(f"Grokking Achieved at Epoch {epoch}!")
                    break

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/history.json', 'w') as f:
        json.dump(history, f)
        
    # Save model
    model_path = f'results/model_{args.dataset}_p{args.p}.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Export to ONNX
    onnx_path = f'results/model_{args.dataset}.onnx'
    dummy_input = torch.randint(0, vocab_size, (1, max_len)).to(device)
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input'], output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"Model exported to {onnx_path}")
        
    print(f"Training finished in {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    train()
