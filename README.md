# Attention Is All You Need


## 01. Input Embedding Layer (embedding.py)

Embedding layer for a Transformer model, inspired by the "Attention Is All You Need" paper. The code defines three key classes: `TokenEmbedding`, `PositionalEncoding`, and `InputEmbedding`.

### Token Embedding

The `TokenEmbedding` class is responsible for converting input tokens into dense vectors of a fixed size. It leverages PyTorch's `nn.Embedding` layer to perform this conversion. The input to this class is a tensor of tokens, and the output is a tensor of token embeddings.

### Positional Encoding

The `PositionalEncoding` class adds positional information to the token embeddings. This is done using a technique called positional encoding, where a unique vector is added to each token embedding to encode its position in the sequence. The output of this class is a tensor of positional embeddings.

### Input Embedding

The `InputEmbedding` class combines the functionalities of the `TokenEmbedding` and `PositionalEncoding` classes. It first computes the token embeddings using the `TokenEmbedding` class and then adds the positional embeddings using the `PositionalEncoding` class. The final output is a tensor of input embeddings that are ready to be fed into the Transformer model.


## 02. Layer Normalization (normalization.py)

### Layer Normalization in PyTorch

Layer normalization, which is a technique used to normalize the activations of a layer in a neural network. Layer normalization can help improve the stability and performance of neural networks, particularly in the context of recurrent neural networks and transformers.

