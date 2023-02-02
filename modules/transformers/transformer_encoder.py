import torch
import torch.nn as nn

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_heads, d_embed, dropout_attn, dropout_proj, device = config.n_heads, config.d_embed, config.dropout_attn, config.dropout_proj, config.device

        assert d_embed % n_heads == 0        
        self.hid_dim = d_embed
        self.n_heads = n_heads
        self.head_dim = d_embed // n_heads
        
        self.fc_q = nn.Linear(d_embed, d_embed)
        self.fc_k = nn.Linear(d_embed, d_embed)
        self.fc_v = nn.Linear(d_embed, d_embed)
        self.proj = nn.Linear(d_embed, d_embed)
        
        self.dropout_attn = nn.Dropout(dropout_attn)
        self.dropout_proj = nn.Dropout(dropout_proj)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, x, mask = None):       
        batch_size = x.shape[0]

        Q,K,V = self.fc_q(x), self.fc_k(x), self.fc_v(x) #[batch size, query len, hid dim
    
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch size, n heads, query len, head dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) 
                
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) /  self.scale  #attention = [batch size, n heads, query len, key len]
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))
        
        attention = self.dropout_attn(torch.softmax(attention, dim = -1)) # Attention Dropout;  attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(attention, V) #x = [batch size, n heads, query len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous() #x = [batch size, query len, n heads, head dim]
        x = x.view(batch_size, -1, self.hid_dim) #x = [batch size, query len, hid dim] (hidden concat of all heads)
        x = self.dropout_proj(self.proj(x)) #x = [batch size, query len, hid dim]

        return x #,attention

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embed = nn.Embedding(config.enc_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.context_len, config.d_embed))
        self.enc_block = nn.ModuleList([EncoderBlock(config) for _ in range(config.enc_blocks)])
        self.dropout_pos = nn.Dropout(config.dropout_pos)
        self.norm = nn.LayerNorm(config.d_embed)

    def forward(self, input, mask=None):
        x = self.tok_embed(input)
        x_pos = self.pos_embed[:, :x.size(1), :]
        x = self.dropout_pos(x + x_pos)
        for layer in self.enc_block:
            x = layer(x, mask)
        return self.norm(x)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        
        self.mhsa = MultiHeadedSelfAttention(config)
        self.pwff = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_embed),
            nn.Dropout(config.dropout_pwff)
        )
        self.ln_mhsa = nn.LayerNorm(config.d_embed)
        self.ln_pwff = nn.LayerNorm(config.d_embed)
        self.dropout = nn.Dropout(config.dropout_res)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.mhsa(self.ln_mhsa(x)))   # Pre-LN
        x = x + self.dropout(self.pwff(self.ln_pwff(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.encoder = Encoder(config)
        self.linear = nn.Linear(config.d_embed, num_classes)

    def forward(self, x, pad_mask=None):
        x = self.encoder(x, pad_mask)
        return  self.linear(torch.mean(x,-2))



@dataclass
class ModelConfig:
    enc_vocab_size: int
    d_embed: int
    d_ff: int
    n_heads: int
    enc_blocks: int
    context_len: int
    dropout_res: float
    dropout_attn: float
    dropout_proj: float
    dropout_pos: float
    dropout_pwff: float
    device: str

num_classes = 2
device = "cuda"
vocab_size = 2048
context_len = 128


config = ModelConfig(enc_vocab_size = vocab_size,
    d_embed = 32,
    d_ff = 4*32,
    n_heads = 4,
    enc_blocks = 1,
    context_len = context_len,
    dropout_res = 0.1,
    dropout_attn = 0.1,
    dropout_proj= 0.1,
    dropout_pos=  0.1,
    dropout_pwff = 0.1,
    device="cuda"
)
        
def make_model(config):
    model = Transformer(config, num_classes).to(device)
    for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model


