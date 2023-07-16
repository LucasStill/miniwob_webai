import torch.nn as nn
import torch

# Try with different output
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        #self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        #self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        #self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()


    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cpu')
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cpu')

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc_1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc(out)
        return out

        # Now split into three output tensors for action_type, ref, and keydown_text
        #action_type_tensor = out[:, 1]
        #ref_tensor = out[:, 1:17]
        #keydown_tensor = out[:, 17:]

        #return action_type_tensor, ref_tensor, keydown_tensor


def test():
    batch_size = 8
    seq_length = 1024
    num_classes = 49
    input_size = 128
    hidden_size = 512
    num_layers = 2

    x = torch.randn(batch_size, seq_length, input_size).to('cpu')
    print(f'input: {x.shape}')
    y = torch.randint(0, num_classes, (batch_size,)).to("cpu")

    # Initialize the model
    model = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length).to('cpu')

    action_type_tensor, ref_tensor, keydown_tensor = model.forward(x)
    print(f'action_type_tensor: {action_type_tensor.shape}, ref_tensor: {ref_tensor.shape}, keydown_tensor: {keydown_tensor.shape}')


from CC_NeT5.language import encode_ref
from torch.utils.data import DataLoader
import torch.optim as optim

def test_old():
    num_classes = 1
    input_size = 1
    hidden_size = 64
    num_layers = 2
    seq_length = 1
    learning_rate = 0.001
    num_epochs = 1000

    model = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    integers = [10, 42, 57, 128, 256, 512, 1024, 2048, 4096, 8192, 1, 7, 21]

    # Convert the integers to tensors
    data = torch.tensor(integers).view(-1, 1, 1).float()

    # Split the data into training and test sets
    train_data = data[:10]
    test_data = data[10:]

    # Normalize the data (optional)
    #mean = train_data.mean()
    #std = train_data.std()
    #train_data = (train_data - mean) / std
    #test_data = (test_data - mean) / std

    for epoch in range(num_epochs):
        model.train()
        outputs = model(train_data)
        loss = criterion(outputs, train_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


def test_training():
    # Small test to teach the model to recode number after encoding them.
    integers = [10, 42, 57, 128, 256, 512, 1024, 2048, 4096, 8192, 1, 7, 21]

    binary_dataset = [(encode_ref(integer)[:8], encode_ref(integer)[8:]) for integer in integers]

    batch_size = 8
    num_batches = len(binary_dataset) // batch_size

    tensor_dataset = []
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        tensor_batch = torch.tensor(binary_dataset[batch_start:batch_end])
        tensor_dataset.append(tensor_batch)

    # Handle the remaining data if it doesn't form a complete batch
    if len(binary_dataset) % batch_size != 0:
        remaining_data = binary_dataset[num_batches * batch_size:]
        tensor_batch = torch.tensor(remaining_data)
        tensor_dataset.append(tensor_batch)

    # Print the sizes of the tensor batches
    for i, tensor_batch in enumerate(tensor_dataset):
        print(f"Tensor Batch {i + 1}: {tensor_batch.size()}")


    train_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

    seq_length = 8
    num_classes = 16
    input_size = 1
    hidden_size = 512
    num_layers = 2

    model = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length).to('cpu')
    print(model)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        for i, tensor_batch in enumerate(tensor_dataset):
            print(f'iter {i}')
            # Forward pass
            outputs = model(tensor_batch.unsqueeze(-1))
            loss = criterion(outputs, tensor_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print training statistics
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Finished Training')


import torch
import torch.nn as nn

class TwoLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TwoLayerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define the LSTM network
# This is the one we'll use
class TwoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TwoLSTM, self).__init__()
        self.hidden_size = hidden_size

        # Note: we use a Linear layer for dimension adjustment.
        # Indeed, the previous multi-modal transformer outputs a 1024 + prev_action length tensor,
        # and here the LSTM layers have a 512 dim where we also need to bypass residual connections.
        # We use the linear layer below to have a matching size with the input/output of these LSTMs
        # in order to concatenate them.
        self.linear = nn.Linear(input_size, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)

        # This linear layer is for the final mapping
        #self.fc = nn.Linear(hidden_size, output_size) #fully connected last layer

    def forward(self, input):
        # Perform dimension adjustment using the linear layer
        adjusted_input = self.linear(input)
        print(f'adjusted_input: {adjusted_input.shape}')

        # Perform the forward pass through each LSTM layer
        output, _hidden_state = self.lstm(adjusted_input)

        # Perform residual connections between LSTM layers
        residual_output = adjusted_input + output

        # Extract the hidden state of the last LSTM layer after residual connections
        last_hidden_state = residual_output[:, -1, :]

        #out = self.fc(last_hidden_state)
        return last_hidden_state

def test_two_lstm():
    input_size = 1024
    hidden_size = 512
    output_size = 49 # Length of the final layer
    model = TwoLSTM(input_size, hidden_size)
    print(model)

    # Generate random input tensor
    batch_size = 8
    sequence_length = 512
    input_data = torch.randn(batch_size, hidden_size, input_size)
    print(f'input_shape: {input_data.shape}')

    #print(input_data.permute(0, 2, 1).shape)

    # Perform a forward pass
    output = model(input_data)

    # Print the shape of the output tensor
    print(f'output: {output.shape}')

    #flattened_tensor = output.view(batch_size, -1)
    #print(f'flattened_tensor: {flattened_tensor.shape}')


def test_new():
    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 1

    model = TwoLayerLSTM(input_size, hidden_size, num_layers, output_size)

    integers = [10, 42, 57, 128, 256, 512, 1024, 2048, 4096, 8192, 1, 7, 21]

    binary_dataset = [(encode_ref(integer)[:8], encode_ref(integer)[8:]) for integer in integers]

    batch_size = 8
    num_batches = len(binary_dataset) // batch_size

    tensor_dataset = []
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        tensor_batch = torch.tensor(binary_dataset[batch_start:batch_end])
        tensor_dataset.append(tensor_batch)

    # Handle the remaining data if it doesn't form a complete batch
    if len(binary_dataset) % batch_size != 0:
        remaining_data = binary_dataset[num_batches * batch_size:]
        tensor_batch = torch.tensor(remaining_data)
        tensor_dataset.append(tensor_batch)

    # Print the sizes of the tensor batches
    for i, tensor_batch in enumerate(tensor_dataset):
        print(f"Tensor Batch {i + 1}: {tensor_batch.size()}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        for i, tensor_batch in enumerate(tensor_dataset):
            print(f'iter {i}')
            # Forward pass
            outputs = model(tensor_batch.unsqueeze(-1))
            loss = criterion(outputs, tensor_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print training statistics
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Finished Training')