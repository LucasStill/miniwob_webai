
import json
import torch

def test():


    # Example tensor
    tensor = torch.randn(3, 260, 180)

    # Convert tensor to a list
    tensor_list = tensor.tolist()

    # Convert list to JSON format
    json_data = json.dumps(tensor_list)

    # Load JSON data as a nested list
    nested_list = json.loads(json_data)

    # Convert nested list to a PyTorch tensor
    retrieved = torch.tensor(nested_list)

    print(tensor == retrieved)
