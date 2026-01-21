import torch
from torch.utils.data import Dataset
import random
import itertools

class ModularAdditionDataset(Dataset):
    def __init__(self, p=113, split='train', train_fraction=0.5, seed=42):
        self.p = p
        self.data = []
        self.labels = []
        
        # Generate all pairs
        pairs = []
        for i in range(p):
            for j in range(p):
                pairs.append((i, j))
        
        # Shuffle and split
        rng = random.Random(seed)
        rng.shuffle(pairs)
        
        split_idx = int(len(pairs) * train_fraction)
        
        if split == 'train':
            self.pairs = pairs[:split_idx]
        else:
            self.pairs = pairs[split_idx:]
            
    def __len__(self):
        return len(self.pairs)
    
    
    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        res = (a + b) % self.p
        # Input format: a, b
        # We output a tensor of shape (2,)
        return torch.tensor([a, b], dtype=torch.long), torch.tensor(res, dtype=torch.long)


class PermutationDataset(Dataset):
    def __init__(self, n=5, split='train', train_fraction=0.5, seed=42):
        self.n = n
        self.split = split
        
        # Generate all permutations of n elements (S_n)
        # S_5 has 120 elements
        elements = list(itertools.permutations(range(n)))
        self.cardinality = len(elements)
        
        self.elem_to_idx = {elem: i for i, elem in enumerate(elements)}
        self.idx_to_elem = {i: elem for i, elem in enumerate(elements)}
        
        # Generate all pairs
        pairs = []
        for i in range(self.cardinality):
            for j in range(self.cardinality):
                pairs.append((i, j))
                
        # Shuffle and split
        rng = random.Random(seed)
        rng.shuffle(pairs)
        
        split_idx = int(len(pairs) * train_fraction)
        
        if split == 'train':
            self.pairs = pairs[:split_idx]
        else:
            self.pairs = pairs[split_idx:]
            
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx_a, idx_b = self.pairs[idx]
        
        perm_a = self.idx_to_elem[idx_a]
        perm_b = self.idx_to_elem[idx_b]
        
        # Composition c = a o b => c[i] = a[b[i]]
        # This corresponds to applying b first, then a.
        # Check standard convention. Usually (ab)(x) = a(b(x)).
        
        res_perm = tuple(perm_a[perm_b[i]] for i in range(self.n))
        
        res_idx = self.elem_to_idx[res_perm]
        
        return torch.tensor([idx_a, idx_b], dtype=torch.long), torch.tensor(res_idx, dtype=torch.long)


class MatrixDataset(Dataset):
    def __init__(self, p=4, n=2, split='train', train_fraction=0.5, seed=42):
        self.p = p
        self.n = n
        self.split = split
        
        # Generate all matrices of size n x n with elements mod p
        # Represented as tuples of length n*n
        elements = list(itertools.product(range(p), repeat=n*n))
        self.cardinality = len(elements)
        
        self.elem_to_idx = {elem: i for i, elem in enumerate(elements)}
        self.idx_to_elem = {i: elem for i, elem in enumerate(elements)}
        
        # Generate pairs
        # Warning: For p=4, n=2, size is 256. Pairs = 65,536. 
        # For larger p or n, this grows very fast.
        
        pairs = []
        # Pre-compute pairs to ensure stable shuffle
        all_indices = list(range(self.cardinality))
        for i in all_indices:
            for j in all_indices:
                pairs.append((i, j))
                
        rng = random.Random(seed)
        rng.shuffle(pairs)
        
        split_idx = int(len(pairs) * train_fraction)
        
        if split == 'train':
            self.pairs = pairs[:split_idx]
        else:
            self.pairs = pairs[split_idx:]
            
    def __len__(self):
        return len(self.pairs)
        
    def matmul(self, A_flat, B_flat):
        # A_flat, B_flat are tuples of size n*n
        # Reshape logically to n x n (row-major)
        # C[i, j] = sum(A[i, k] * B[k, j]) mod p
        
        C = []
        for i in range(self.n):
            for j in range(self.n):
                val = 0
                for k in range(self.n):
                    # A[i, k] is at index i*n + k
                    a_val = A_flat[i * self.n + k]
                    # B[k, j] is at index k*n + j
                    b_val = B_flat[k * self.n + j]
                    val += a_val * b_val
                C.append(val % self.p)
        return tuple(C)
    
    def __getitem__(self, idx):
        idx_a, idx_b = self.pairs[idx]
        mat_a = self.idx_to_elem[idx_a]
        mat_b = self.idx_to_elem[idx_b]
        
        res_mat = self.matmul(mat_a, mat_b)
        res_idx = self.elem_to_idx[res_mat]
        
        return torch.tensor([idx_a, idx_b], dtype=torch.long), torch.tensor(res_idx, dtype=torch.long)



