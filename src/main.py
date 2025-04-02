import tiktoken
import torch
from torch.utils.data import DataLoader

from attention import MultiHeadAttention
from gpt_dataset import GPTDatasetV1


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
    torch.manual_seed(123)

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

    batch = torch.stack((inputs, inputs), dim=0)

    _, context_length, d_in = batch.shape
    d_out = 2

    mha = MultiHeadAttention(
        d_in, d_out, context_length, 0.0, num_heads=2)

    context_vecs = mha(batch)

    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)


if __name__ == "__main__":
    main()
