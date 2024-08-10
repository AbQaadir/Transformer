import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):

    """
    Compute the token embeddings. for each token in the input sequence.
    
    Args:
        vocab_size (int): vocab size
        embed_size (int): embedding size
        
    Inputs:
        tokens: int tensor with shape of [batch_size, seq_len]
        
    Returns:
        embeddings: float tensor with shape of [batch_size, seq_len, embed_size]
        
    """
    
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.embed_size)
    

class PositionalEncoding(nn.Module):
    
    """
    Inject some information about the relative or absolute position of the tokens in the sequence.
    
    Args:
        embed_size (int): embedding size]
        dropout (float): dropout rate
        
        
    Inputs:
        embeddings: float tensor with shape of [batch_size, seq_len, embed_size]
        
    Returns:
        embeddings: float tensor with shape of [batch_size, seq_len, embed_size] 
        
    """
    
    def __init__(self, embed_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, embeddings):
        embeddings = embeddings + self.pe[:, :embeddings.size(1)]
        return self.dropout(embeddings)
    
    
class InputEmbedding(nn.Module):
    
    """
    Compute the input embeddings by summing the token embeddings and positional encodings.
    
    Args:
        vocab_size (int): vocab size
        embed_size (int): embedding size
        dropout (float): dropout rate
        
    Inputs:
        tokens: int tensor with shape of [batch_size, seq_len]
        
    Returns:
        embeddings: float tensor with shape of [batch_size, seq_len, embed_size]
        
    """
    
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super(InputEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, dropout)
        
    def forward(self, tokens):
        return self.positional_encoding(self.token_embedding(tokens))
    
    
    
if __name__ == "__main__":
    
    import tiktoken    
    
    # Load the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Tokenizer: {tokenizer}")
    
    # Example text
    text = "Hello, how are you today?"
    print(f"Text: {text}")
    
    # Tokenize the text
    tokens = tokenizer.encode(text)
    print(f"Tokens: {tokens}")
    
    tokens_tensor = torch.tensor([tokens])
    
    # Test InputEmbedding
    input_embedding = InputEmbedding(tokenizer.max_token_value + 1, 50)
    embeddings = input_embedding(tokens_tensor)
    print(f"Input embeddings: {embeddings}")
    
    print("All tests pass")