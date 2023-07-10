from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Return a ranked list of closest embeddings from out embedding matrix
def find_closest_embeddings(input_embedding, embedding_matrix):
    """
    Finds the index of the closest match to the input embedding based on cosine similarity.

    Args:
        input_embedding (np.ndarray): Array of shape (8,) representing the input embedding.
        embedding_matrix (np.ndarray): Array of shape (10, 8) representing the embedding matrix.

    Returns:
        int: Index of the closest match to the input embedding.
    """
    # Normalize input embedding for cosine similarity
    input_embedding_norm = input_embedding / np.linalg.norm(input_embedding)

    # Normalize embedding matrix for cosine similarity
    embedding_matrix_norm = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)

    # Calculate cosine similarity between input embedding and embedding matrix
    similarity_scores = cosine_similarity(input_embedding_norm.reshape(1, -1), embedding_matrix_norm)

    # Find the index of the closest match
    #closest_index = np.argmax(similarity_scores)

    # List of closest similarities
    return similarity_scores


import torch
import torch.nn as nn

# Embedding Function
class EmbeddingFunction(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingFunction, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_tokens):
        embedded = self.embedding(input_tokens)
        return embedded

    # Provide a tensor and de-embbeds it to retrieve the correct index
    def get_embedding_index(self, x):
        results = torch.where(torch.sum((self.embedding.weight == x), axis=1))
        if len(results[0]) == len(x):
            return None
        else:
            return results[0][0]


def embeddings2tokens(embeddings, embedding_fn):
    embedding_matrix = embedding_fn.embedding.weight.detach().numpy()

    embedding_tensor = embeddings  # Assuming original_tensor is your tensor of size 512
    embedding_size = 64

    num_embeddings = embeddings.shape[0] // embedding_size  # Number of smaller tensors of size 64

    tokens = []
    closest_indices_list = []

    for i in range(num_embeddings):
        start_index = i * embedding_size
        end_index = (i + 1) * embedding_size
        embedding = embedding_tensor[start_index:end_index]
        # Get list of closest embeddings
        closest_indexes = find_closest_embeddings(embedding, embedding_matrix)
        # Get closest_index embedding
        closest_index = np.argmax(closest_indexes)
        # Token, to put into ITOS
        tokens.append(closest_index)
        closest_indices_list.append(closest_indexes)

    return tokens, closest_indices_list


class VocabManagement:
    def __init__(self, vocab_path='vocab.txt'):
        stoi = {}
        itos = {}
        self.padding_char = '<PAD>'
        self.special_characters = ['.', ',', '#', ':', '-', '/', '(', ')', 'https://', '@', '&', '"', "'", '!', '?',
                                   ';', '+', '=', '*', '$', '€', '*', '`']

        with open(vocab_path, 'r') as file:
            for index, line in enumerate(file):
                line = line.strip()
                stoi[line] = index
                itos[index] = line

        stoi[' '] = stoi['']
        itos[stoi[' ']] = ' '
        self.stoi = stoi
        self.stoi[self.padding_char] = len(stoi.keys())-1  # Add PADDING character
        # We do not need itos as we don't implement a de-tokinezing function.
        self.itos = itos
        self.itos[len(stoi.keys()) - 2] = self.padding_char  # Add PADDING character
        print(f'pad: {len(stoi.keys()) - 2}')


if __name__ == '__main__':
    tok = VocabManagement()
    print(set(tok.stoi.values()))
    print(set(tok.itos.keys()))

    print(f'padding: {tok.stoi[tok.padding_char]}')
    print(len(tok.itos))
    print(tok.itos[1590])