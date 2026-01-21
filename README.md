# Grokking Experiment

This project is a test for investigating the "grokking" phenomenon in neural networks. 
It involves training small transformers on algorithmic datasets (like modular addition) to observe the transition from memorization to generalization.

## Installation

1.  Clone this repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Training

The main training script is `src/train.py`. You can configure the training process using the `config.json` file or command-line arguments.

### Using Key Arguments

-   `--dataset`: Choose the dataset (`modular`, `permutation`, `matrix`).
-   `--epochs`: Number of training epochs.
-   `--p`: Modulus for modular addition (default: 113).
-   `--lr`: Learning rate.
-   `--weight_decay`: Weight decay (crucial for grokking).

### Examples

**Train with default settings (Modular Addition):**
```bash
python src/train.py
```

**Train on Permutation Group $S_5$:**
```bash
python src/train.py --dataset permutation
```

**Train on Matrix Multiplication ($GL_2(\mathbb{Z}_4)$):**
```bash
python src/train.py --dataset matrix --matrix_p 4 --matrix_n 2
```

## Data

The project supports three types of algorithmic datasets, defined in `src/data.py`:

1.  **Modular Addition (`modular`)**:
    -   Task: Calculate $(a + b) \pmod p$.
    -   Input: Two integers $a, b$.
    -   Output: The result of the modular addition.
    -   Complexity: Controlled by prime $p$.

2.  **Permutation Group (`permutation`)**:
    -   Task: Calculate the composition of two permutations $\sigma \circ \tau$.
    -   Input: Indices of two permutations from $S_n$.
    -   Output: Index of the resulting permutation.
    -   Default: $n=5$ ($S_5$), size 120.

3.  **Matrix Multiplication (`matrix`)**:
    -   Task: Multiply two square matrices $A \times B \pmod p$.
    -   Input: Indices of two flattened matrices.
    -   Output: Index of the resulting matrix.
    -   Parameters: `matrix_n` (dimension), `matrix_p` (modulus).

## Analysis & Plotting

After training, results are saved in `results/history.json`. You can visualize the training dynamics using `src/plot.py`.

```bash
python src/plot.py
```

This will generate `results/grokking_plot.png`, showing Training vs. Validation Accuracy over time. The x-axis is log-scaled to better visualize the "grokking" phase transition.
