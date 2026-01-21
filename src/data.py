import torch
from torch.utils.data import Dataset
import random
import itertools
import datetime

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




class DateDataset(Dataset):
    def __init__(self, start_year=1900, end_year=2100, split='train', train_fraction=0.5, seed=42):
        self.data = []
        
        # Generate all dates in range
        dates = []
        start_date = datetime.date(start_year, 1, 1)
        end_date = datetime.date(end_year, 12, 31)
        delta = end_date - start_date
        
        for i in range(delta.days + 1):
            day = start_date + datetime.timedelta(days=i)
            # Input: (Year, Month, Day)
            # We normalize year to be generally handleable. Let's keep raw year for now or offset.
            # Keeping raw year (approx 2000) might be large for embeddings if not careful? 
            # Actually, standard transformer embeddings won't like vocab size 2100 if we use simple embedding.
            # But the existing `SimpleTransformer` uses `vocab_size` and `nn.Embedding`.
            # So input tokens must be < vocab_size.
            # Years 1900-2100 -> indices 0..200? Or just treat Year, Month, Day as independent tokens?
            # Issue: Year 2000 is index 2000. Vocab size must be > 2100. This is fine.
            
            # Label: Monday=0, Sunday=6
            dates.append(((day.year, day.month, day.day), day.weekday()))
            
        self.cardinality = len(dates) # Approx 73000
        
        rng = random.Random(seed)
        rng.shuffle(dates)
        
        split_idx = int(len(dates) * train_fraction)
        
        if split == 'train':
            self.pairs = dates[:split_idx]
        else:
            self.pairs = dates[split_idx:]
            
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        inp, label = self.pairs[idx]
        # inp is (y, m, d)
        return torch.tensor(inp, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class TicTacToeDataset(Dataset):
    def __init__(self, split='train', train_fraction=0.5, seed=42):
        # Generate all 3^9 = 19683 states
        # check winner for each
        
        self.states = []
        
        # 0=Empty, 1=X, 2=O
        for board_flat in itertools.product(range(3), repeat=9):
            # Check winner
            winner = self.check_winner(board_flat)
            self.states.append((board_flat, winner))
            
        rng = random.Random(seed)
        rng.shuffle(self.states)
        
        split_idx = int(len(self.states) * train_fraction)
        
        if split == 'train':
            self.pairs = self.states[:split_idx]
        else:
            self.pairs = self.states[split_idx:]
            
    def check_winner(self, board):
        # board is tuple of 9
        # Indices:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        wins = [
            (0,1,2), (3,4,5), (6,7,8), # rows
            (0,3,6), (1,4,7), (2,5,8), # cols
            (0,4,8), (2,4,6)           # diags
        ]
        
        x_wins = False
        o_wins = False
        
        for a,b,c in wins:
            if board[a] == board[b] == board[c]:
                if board[a] == 1: x_wins = True
                if board[a] == 2: o_wins = True
                
        # If both win -> Invalid state usually, map to 0 (Draw/Invalid) to avoid confusion? 
        # Or priority? Let's say Invalid/Draw is 0.
        if x_wins and not o_wins: return 1
        if o_wins and not x_wins: return 2
        return 0
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        board, label = self.pairs[idx]
        return torch.tensor(board, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class GameOfLifeDataset(Dataset):
    def __init__(self, size=5, samples_per_core=20, split='train', train_fraction=0.5, seed=42):
        # Size 5x5 grid
        # We generate data by iterating all 2^9 (512) center-3x3 patterns (which determine the center cell next state)
        # And for each core pattern, we generate 'samples_per_core' random border configurations.
        # This teaches the model that ONLY the local neighborhood matters.
        
        self.data = []
        
        # 5x5 grid indices:
        # 0  1  2  3  4
        # 5  6  7  8  9
        # 10 11 12 13 14  <-- Center is 12. Neighbors are 6,7,8, 11,13, 16,17,18
        # 15 16 17 18 19
        # 20 21 22 23 24
        
        # Core neighbors for center cell (idx 12)
        # (row, col) from (0,0) to (4,4)
        # Neighbors of (2,2): (1,1)..(3,3) excluding (2,2)
        # Offsets in flat array:
        # row 1: 5+1=6, 5+2=7, 5+3=8
        # row 2: 10+1=11,      10+3=13
        # row 3: 15+1=16, 15+2=17, 15+3=18
        
        # Wait, the rule depends on the center cell itself too.
        # So the "core" is the full 3x3 block centered at 2,2.
        # Indices: 6,7,8, 11,12,13, 16,17,18.
        
        core_indices = [6,7,8, 11,12,13, 16,17,18]
        other_indices = [i for i in range(25) if i not in core_indices] # 16 cells in border
        
        rng = random.Random(seed)
        
        dataset_list = []
        
        # Iterate all 2^9 = 512 core configurations
        for core_vals in itertools.product(range(2), repeat=9):
            # Calculate label (next state of center)
            # Map core_vals to their positions
            # core_vals corresponds to core_indices in order
            
            # Extract center status and neighbor count
            # Center is at index 4 in core_indices (12) -> index 4 in core_vals
            curr_center = core_vals[4]
            others = core_vals[:4] + core_vals[5:]
            alive_neighbors = sum(others)
            
            # GoL Label Rule
            # 1. Any live cell with < 2 live neighbours dies.
            # 2. Any live cell with 2 or 3 live neighbours lives.
            # 3. Any live cell with > 3 live neighbours dies.
            # 4. Any dead cell with exactly 3 live neighbours becomes a live cell.
            
            next_state = 0
            if curr_center == 1:
                if alive_neighbors in [2, 3]:
                    next_state = 1
                else:
                    next_state = 0
            else:
                if alive_neighbors == 3:
                    next_state = 1
                else:
                    next_state = 0
                    
            # Generate noisy borders
            for _ in range(samples_per_core):
                # Random border
                grid = [0] * 25
                # Fill core
                for k, idx in enumerate(core_indices):
                    grid[idx] = core_vals[k]
                # Fill border randomly
                for idx in other_indices:
                    grid[idx] = rng.choice([0, 1])
                    
                dataset_list.append((tuple(grid), next_state))
                
        rng.shuffle(dataset_list)
        
        split_idx = int(len(dataset_list) * train_fraction)
        
        if split == 'train':
            self.pairs = dataset_list[:split_idx]
        else:
            self.pairs = dataset_list[split_idx:]
            
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        grid, label = self.pairs[idx]
        return torch.tensor(grid, dtype=torch.long), torch.tensor(label, dtype=torch.long)
