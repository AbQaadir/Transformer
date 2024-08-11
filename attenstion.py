import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module with scaled dot-product attention and output projection.
    
    Args:
        d_model: int, the dimension of the input
        h: int, the number of heads
        dropout: float, the dropout
        
    Inputs:
        q: torch.Tensor, the query tensor with shape (batch_size, seq_length, d_model)
        k: torch.Tensor, the key tensor with shape (batch_size, seq_length, d_model)
        v: torch.Tensor, the value tensor with shape (batch_size, seq_length, d_model)
        mask: torch.Tensor, the mask tensor with shape (batch_size, seq_length, seq_length)
        
    Outputs:
        output: torch.Tensor, the output tensor with shape (batch_size, seq_length, d_model)
        weights: torch.Tensor, the attention weights with shape (batch_size, h, seq_length, seq_length)
    """

    def __init__(self, d_model: int, h: int, dropout: float):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model # the dimension of the input
        self.h = h # the number of heads
        assert d_model % h == 0, "d_model must be divisible by h" # check if d_model is divisible by h

        self.d_k = d_model // h # the dimension of the key and value

        self.W_q = nn.Linear(d_model, d_model) # the query projection
        self.W_k = nn.Linear(d_model, d_model) # the key projection
        self.W_v = nn.Linear(d_model, d_model) # the value projection

        self.W_o = nn.Linear(d_model, d_model) # the output projection

        self.dropout = nn.Dropout(dropout) # the dropout
        
        @staticmethod
        def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, d_k: int, dropout: nn.Dropout, mask: torch.Tensor = None) -> torch.Tensor:
            
            self_attention = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())
            
            if mask is not None:
                self_attention = self_attention.masked_fill(mask == 0, -1e9)
            
            attention = F.softmax(self_attention, dim=-1)
            attention = dropout(attention)
            output = torch.matmul(attention, v)
            return output
                    
            
            
            
        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
            batch_size = q.size(0)
            
            query = self.W_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) # (batch_size, h, seq_length, d_k)
            key = self.W_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            value = self.W_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

            
        
        

    
    