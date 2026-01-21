import matplotlib.pyplot as plt
import json
import argparse

def plot_results():
    with open('results/history.json', 'r') as f:
        history = json.load(f)
        
    plt.figure(figsize=(10, 6))
    plt.plot(history['step'], history['train_acc'], label='Train Accuracy')
    plt.plot(history['step'], history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Optimization Steps')
    plt.ylabel('Accuracy (%)')
    plt.title('Grokking: Modular Addition')
    plt.legend()
    plt.grid(True)
    plt.xscale('log') # Grokking often looks best on log scale
    plt.savefig('results/grokking_plot.png')
    plt.show()

if __name__ == '__main__':
    plot_results()
