import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.token_embed = TokenEmbed(model_config)
        self.pos_embed = PosEmbed(model_config)
        self.layers = nn.ModuleList([TransformerLayer(model_config) for _ in range(model_config['num_layers'])])
        self.token_unembed = TokenUnembed(model_config)

    def forward(self, x):
        x = self.token_embed(x)
        x = x + self.pos_embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.token_unembed(x)

class TransformerLayer(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.attn = AttentionBlock(model_config)
        self.mlp = MLP(model_config)
    
    def forward(self, x):
        x +=self.attn(x)
        x += self.mlp(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.key = nn.Linear(model_config['hidden_size'], model_config['key_size'])
        self.query = nn.Linear(model_config['hidden_size'], model_config['key_size'])
        self.value = nn.Linear(model_config['hidden_size'], model_config['key_size'])
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(model_config['key_size'], model_config['hidden_size'])
        
    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        attn_mat = self.softmax(torch.einsum('ij,kj->ik', q, k))
        weighted_vals = torch.einsum('ij,jk->ik',attn_mat, v)
        return self.linear(weighted_vals)

class MLP(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.linear_1 = nn.Linear(model_config['hidden_size'], model_config['mlp_hidden_size'])
        self.linear_2 = nn.Linear(model_config['mlp_hidden_size'], model_config['hidden_size'])
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x
    
class PosEmbed(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.embeddings = nn.Parameter(
            torch.empty(model_config['max_seq_len'], model_config['hidden_size'])
        )

    def forward(self, x):
        # x: (seq_len)
        # self.embed: (vocab_size, attn_hidden_size)
        return self.embeddings[:x.shape[0], :] # (seq_len, attn_hidden_size)
    
class TokenEmbed(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.embeddings = nn.Parameter(
            torch.empty(model_config['vocab_size'], model_config['hidden_size'])
        )

    def forward(self, x):
        # x: (seq_len)
        # self.embed: (vocab_size, attn_hidden_size)
        return self.embeddings[x, :] # (seq_len, attn_hidden_size)
    
class TokenUnembed(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.unembed = torch.eye(model_config['hidden_size'], model_config['vocab_size_out'])

    def forward(self, x):
        # x: (seq_len, hidden_dim)
        # out: (seq_len, vocab_size_out)
        return torch.matmul(x, self.unembed)
