import torch
import torch.nn as nn
from torch.nn import functional as F
# from tqdm import tqdm


# Hyperparameters : Global variables
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32  #  C
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


# Simple Bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of intergers
        tok_embd = self.token_embedding_table(idx)  # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_embd + pos_embd  # (B, T, C)
        logits = self.lm_head(tok_embd)  # (B, T, vocal_size)
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
            # get predictions
            logits, loss = self(idx)
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
