import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


class SelfAttentionV2(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values

        return context_vec


    # Constants
BATCH_SIZE = 8
MAX_LENGTH = 4
STRIDE = 4


def input_embedding_pipeline(inputs, max_length=MAX_LENGTH, output_dim=256, vocab_size=50257):
    output_dim = 256
    vocab_size = 50257

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    positional_embedding_layer = torch.nn.Embedding(max_length, output_dim)

    token_embeddings = token_embedding_layer(inputs)
    positional_embeddings = positional_embedding_layer(
        torch.arange(max_length))

    return token_embeddings + positional_embeddings


def main():
    torch.manual_seed(789)

    # with open("./data/the-verdict.txt", "r", encoding="utf-8") as f:
    #    raw_text = f.read()

    # dataloader = create_dataloader_v1(
    #    raw_text, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, stride=STRIDE, shuffle=False)
    # data_iter = iter(dataloader)
    # inputs, targets = next(data_iter)

    # input_embeddings = input_embedding_pipeline(inputs)

    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55]]  # step     (x^6)
    )

    d_in = inputs.shape[1]
    d_out = 2

    sa_v1 = SelfAttentionV2(d_in, d_out)

    print(sa_v1(inputs))


if __name__ == "__main__":
    main()
