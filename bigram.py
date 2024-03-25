import torch
import torch.nn as nn
from torch.nn import functional as F
# from tqdm import tqdm


# Hyperparameters : Global variables
batch_size = 64  # how many independent sequences wil we process in prallel
block_size = 256   # what is the maximum context length for preditction
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384  #  C  (must be a multiple of n_head)
n_head = 6
n_layer = 6
dropout = 0.2
#----------------

torch.manual_seed(42)

# read data
with open(r'./assets/contes.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get all unique characters that occur
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Encoder and decoder funcitons
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoding part
decode = lambda l: ''.join([itos[i] for i in l])  # decoding part

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('trill', torch.tril(torch.ones(block_size, block_size)))  # register_buffer to create something different than an attribut

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.trill[:T, :T] == 0, float('-inf'))  # (B, T, C)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple head of self-attention in parallel"""

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Feedforward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # projection layer
            nn.Dropout(dropout),
            )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = Feedforward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# Simple Bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        # self.sa_head = MultiHeadAttention(4, n_embd // 4)  # i.e. 4 heads pf 8-dimensional self-attention
        # self.ffwd = Feedforward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of intergers
        tok_embd = self.token_embedding_table(idx)  # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_embd + pos_embd  # (B, T, C)
        # x = self.sa_head(x)  # Apply one head of self_attention (B, T, C)
        # x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocal_size)
        # B => Number of batch
        # T => The size of the context
        # C => Number od Channel, i.e. the size of the vocabulary
        if targets is None:
            loss = None
        else:
            # For the cross_entropy the channel C must be the second one
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            # focus only in the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx
    

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # get the gradients for all parameters
    optimizer.step()  # uptdate parameters


context = torch.zeros((1,1), dtype=torch.long)
print(decode(list(m.generate(context, max_new_tokens=500)[0].numpy())))
