import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    """
    Construct a layernorm module (See citation for details)
    
    Args:
        features (int): the size of the layer
        eps (float): the epsilon value to avoid division by zero
        
    Inputs:
        x: float tensor with shape of [batch_size, seq_len, features]
        
    Returns:
        x: float tensor with shape of [batch_size, seq_len, features]
        
    """
    
    def __init__(self, features, eps=1e-6):
        super(LayerNormalization, self).__init__()
        
        self.alpha = nn.Parameter(torch.ones(features)) # weight
        
        self.bias = nn.Parameter(torch.zeros(features)) # bias
        
        self.eps = eps # epsilon
        
    def forward(self, x):
        
        mean = x.mean(-1, keepdim=True) # mean
        std = x.std(-1, keepdim=True) # standard deviation
        return self.alpha * (x - mean) / (std + self.eps) + self.bias # normalization
    
    
if __name__ == "__main__":
    
    # Define a layernorm module
    layernorm = LayerNormalization(64)
    print(f"LayerNormalization: {layernorm}")
    
    x = torch.randn(4, 5, 64)
    print(f"Input tensor: {x}")
    
    y = layernorm(x)
    print(f"Output tensor: {y}")
