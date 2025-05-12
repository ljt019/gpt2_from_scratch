import tiktoken
import torch
from torch.utils.data import DataLoader

from attention import MultiHeadAttention
from gpt_dataset import GPTDatasetV1
from model import GptModel, TransformerBlock


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


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        probas = torch.softmax(logits, dim=-1)

        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main():
    torch.manual_seed(123)
    torch.set_printoptions(sci_mode=False)

    # with open("./data/the-verdict.txt", "r", encoding="utf-8") as f:
    #    raw_text = f.read()

    # dataloader = create_dataloader_v1(
    #    raw_text, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, stride=STRIDE, shuffle=False)
    # data_iter = iter(dataloader)
    # inputs, targets = next(data_iter)

    # input_embeddings = input_embedding_pipeline(inputs)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    model = GptModel(GPT_CONFIG_124M)

    start_context = "Hello, I am"

    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    model.eval()

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(decoded_text)


if __name__ == "__main__":
    main()
