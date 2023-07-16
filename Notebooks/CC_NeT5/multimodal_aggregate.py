import torch
import torch.nn as nn

class MultiModalTransformer(nn.Module):
    def __init__(self, num_layers=8, num_heads=8, hidden_dim=512, dropout=0.1, input_dim=140):
        super(MultiModalTransformer, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        # Removed the final layer to have dim (1024, 8, 512).
        # This is perhaps too much, so in case we can reduce it with a linear layer
        #self.fc = nn.Linear(hidden_dim, 128)

    def forward(self, input):
        # Calculate the mask for padding
        mask = (input.sum(dim=-1) != 0)  # Remove the transpose operation

        # Apply linear embedding
        embedded = self.embedding(input)

        # Permute dimensions for transformer input (sequence_length, batch_size, hidden_dim)
        embedded = embedded.permute(1, 0, 2)

        # Apply transformer layers
        for layer in self.transformer_layers:
            embedded = layer(embedded, src_key_padding_mask=mask)

        # Permute dimensions for linear layer (batch_size, sequence_length, hidden_dim)
        embedded = embedded.permute(1, 0, 2)

        # Apply final linear layer
        #output = self.fc(embedded)

        return embedded




def test():
    # Define the input parameters
    batch_size = 8
    sequence_length = 1024
    input_dim = 140

    # Generate random input data
    input_data = torch.randn(batch_size, sequence_length, input_dim)
    print(f'shape input: {input_data.shape}')

    # Initialize the MultiModalTransformer model
    num_layers = 4
    num_heads = 4
    hidden_dim = 512
    dropout = 0.1
    model = MultiModalTransformer()

    print(model)

    # Pass the input data through the model
    output = model(input_data)

    # Print the output shape
    print("Output shape:", output.shape)

    print(output)


import torch
import torch.nn as nn
from torch.nn import Transformer

class TransformerNetwork(nn.Module):
    def __init__(self):
        super(TransformerNetwork, self).__init__()

        self.embedding_dim = 512
        self.num_layers = 8
        self.num_heads = 8

        self.embedding = nn.Linear(140, self.embedding_dim)
        self.transformer = Transformer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers
        )
        #self.output_layer = nn.Linear(self.embedding_dim, 1)

    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)

        embedded = self.embedding(input_tensor)
        embedded = embedded.permute(1, 0, 2)

        output = self.transformer(embedded, embedded)
        output = output.permute(1, 0, 2)
        #output = self.output_layer(output)

        return output.squeeze(2)

def test3():
    # Create an instance of the TransformerNetwork
    network = TransformerNetwork()

    # Generate dummy input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1024, 140)

    # Forward pass through the network
    output = network(input_tensor)
    print(output.shape)



