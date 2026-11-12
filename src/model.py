import torch
from torch import nn

class MutationMLP(nn.Module):
    """Small regression network; no pretrained weights and no antigen identity input."""
    def __init__(self, input_dim, hidden=64, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)

def predict_array(model, x):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
