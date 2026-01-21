import torch
import torch.nn as nn
import math

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers, max_len=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x)
        # We only care about the last token representation for prediction, or maybe we pool?
        # Standard approach for grokking addition is often to take the last token or just flatten.
        # However, here we input [a, b] and want to predict c.
        # Let's effectively treat it as a sequence task where we predict the next token, 
        # but here we just want the output for the whole sequence.
        # We'll take the representation of the last token.
        x = x[:, -1, :] 
        return self.fc_out(x)
