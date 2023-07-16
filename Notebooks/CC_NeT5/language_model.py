
import torch
import torch.nn as nn


import torch
import torch.nn as nn

class LanguageModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256, num_heads=4):
        super(LanguageModel, self).__init__()
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads)
        self.cross_attention = nn.MultiheadAttention(input_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.embedding = nn.Linear(input_dim, 512)

    def forward(self, input_tensor):
        # Self-attention
        attended_tensor, _ = self.self_attention(input_tensor, input_tensor, input_tensor)
        attended_tensor = attended_tensor + input_tensor  # Residual connection

        # Cross-attention
        cross_attended_tensor, _ = self.cross_attention(attended_tensor, attended_tensor, attended_tensor)
        cross_attended_tensor = cross_attended_tensor + attended_tensor  # Residual connection

        # Feed-forward
        output_tensor = self.feed_forward(cross_attended_tensor)
        output_tensor = output_tensor + cross_attended_tensor  # Residual connection

        # Embedding
        print(f'output_tensor: {output_tensor.shape}')
        embedding = self.embedding(output_tensor.mean(dim=1))  # Average pooling over sequence dimension

        return output_tensor

class LanguageModel2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(LanguageModel2, self).__init__()

        # Self-attention mechanism
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

        # Feed-forward neural network
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Layer normalization: Pre-Norm and Post-Norm as it better stabilizes the gradient flow
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Apply self-attention mechanism
        attention_output, _ = self.cross_attention(x, x, x)
        x = self.layer_norm1(x + attention_output)

        # Apply feed-forward neural network
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)

        return x


import torch
import torch.nn as nn

class CrossAttentionModelLanguage(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=140, num_heads=4):
        super(CrossAttentionModelLanguage, self).__init__()

        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, input):
        # Reshape the input to (sequence_length, batch_size, input_dim)
        input = input.permute(1, 0, 2)

        # Apply cross-attention
        output, _ = self.attention(input, input, input)

        # Reshape the output to (batch_size, sequence_length, input_dim)
        output = output.permute(1, 0, 2)

        # Apply linear transformation
        output = self.linear(output)

        return output


def test_cross_attention():
    batch_size = 8
    input_dim = 64

    input_tensor = torch.randn(batch_size, 512, input_dim)
    print(input_tensor.shape)
    transformer = CrossAttentionModelLanguage()
    output_embedding = transformer(input_tensor)

    print(output_embedding.shape)  # Output: torch.Size([10, 512])


def test():
    # Example usage
    batch_size = 10
    input_dim = 64

    input_tensor = torch.randn(batch_size, 512, input_dim)
    print(input_tensor.shape)
    transformer = LanguageModel()
    output_embedding = transformer(input_tensor)

    print(output_embedding.shape)  # Output: torch.Size([10, 512])


import torch
import torch.nn as nn


class CrossAttentionTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CrossAttentionTransformer, self).__init__()

        self.encoder = nn.Linear(input_dim, output_dim)
        self.decoder = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Compute self-attention
        encoded = self.encoder(x)
        query = self.decoder(x)
        attn_weights = torch.bmm(
            query.unsqueeze(1), encoded.unsqueeze(2)
        )  # Swap dimensions for bmm
        attn_weights = self.softmax(attn_weights)

        # Apply cross attention
        cross_attn = torch.bmm(attn_weights, encoded.unsqueeze(2))
        output = cross_attn.squeeze(2)

        return output


def test4():
    # Create an instance of the CrossAttentionTransformer
    model = CrossAttentionTransformer(input_dim=552, output_dim=512)

    # Generate a random input tensor
    batch_size = 8
    input_tensor = torch.randn(batch_size, 552)

    # Pass the input tensor through the model
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Should be (batch_size, 512)
