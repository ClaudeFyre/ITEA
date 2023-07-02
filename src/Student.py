import torch
from torch_geometric.nn import SAGEConv

class StudentModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(StudentModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

    def train_model(self, data, device, epochs=100):
        model = self.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.MSELoss()  # or whatever loss function suits your problem

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data.x.to(device), data.edge_index.to(device))
            loss = criterion(out, data.y.to(device))  # assuming data.y is the target
            loss.backward()
            optimizer.step()
        return model
