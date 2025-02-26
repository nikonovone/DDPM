import torch


def get_time_embedding(seq_len, d_model, device="cpu"):
    # Create position indices: [0, 1, 2, ..., seq_len-1]
    positions = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)

    # Create dimension indices: [0, 2, 4, ..., d_model-2] for sine
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=device)
        * -(torch.log(torch.tensor(10000.0)) / d_model),
    )

    # Create empty encoding matrix
    pe = torch.zeros(seq_len, d_model, device=device)

    # Compute sine for even dimensions and cosine for odd dimensions
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe
