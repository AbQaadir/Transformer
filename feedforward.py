import torch
import torch.nn as nn
import torch.nn.functional as F



class FeedForward(nn.Module):
    
    """
    A simple feedforward neural network with 1 hidden layer.
    
    Args:
        input_dim (int): the size of the input features
        hidden_dim (int): the output size of the first Linear layer
        output_dim (int): the output size of the second Linear layer
        
    Inputs:
        x_in (torch.Tensor): an input data tensor with shape [batch_size, input_dim]
        apply_softmax (bool): a flag for the softmax activation
        
    Returns:
        the resulting tensor tensor with shape [batch_size, output_dim]
    """
    
    
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        
        super(FeedForward, self).__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x_in: torch.Tensor, apply_softmax: bool = False) -> torch.Tensor:
            
        x = self.fc2(self.dropout(F.relu(self.fc1(x_in))))
        
        if apply_softmax:
            x = F.softmax(x, dim=1)
            
        return x
          
    
    
if __name__ == "__main__":
    
    # Define a feedforward neural network
    feedforward = FeedForward(512, 2048, 0.1)
    print(f"FeedForward: {feedforward}")
    
    x_in = torch.randn(64, 512)
    print(f"Input tensor: {x_in}")
    
    y_out = feedforward(x_in)
    print(f"Output tensor: {y_out}")
    
    y_out = feedforward(x_in, apply_softmax=True)
    print(f"Output tensor with softmax: {y_out}")