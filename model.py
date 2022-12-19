import torch
from torchinfo import summary

class DropPercentModel(torch.nn.Module):
    def __init__(self) -> None:
        super(DropPercentModel, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.tanh = torch.nn.Tanh()
        self.ff1 = torch.nn.Linear(16, 32)
        self.ff2 = torch.nn.Linear(32, 64)
        self.ff3 = torch.nn.Linear(64, 128)
        self.ff4 = torch.nn.Linear(128, 128)
        self.ff5 = torch.nn.Linear(128, 64)
        self.ff6 = torch.nn.Linear(64, 32)
        self.ff7 = torch.nn.Linear(32, 16)
        self.ff8 = torch.nn.Linear(16, 1)
        # self.dropout = torch.nn.Dropout(p=0.3)
        self.bn = torch.nn.BatchNorm1d(5)
        self.lstm = torch.nn.LSTM(5, 16, batch_first=True)

    def forward(self, x):
        # (N, L, C) => (N, C, L)
        # x = x.transpose(1, 2)
        # x.data = self.bn(x.data)
        # x = x.transpose(1, 2)

        if isinstance(x, tuple):
          x_padded = x[0]
          lens_x_padded = x[1]
          x = torch.nn.utils.rnn.pack_padded_sequence(x_padded, lens_x_padded,
              batch_first=True,
              enforce_sorted=False)

        # x_padded = x[0]
        # lens_x_padded = x[1]
        # x = torch.nn.utils.rnn.pack_padded_sequence(x_padded, lens_x_padded,
        #     batch_first=True,
        #     enforce_sorted=False)

        _, (hn, cn) = self.lstm(x)
        x = hn.squeeze()
        x = self.relu(self.ff1(x))
        x = self.relu(self.ff2(x))
        x = self.relu(self.ff3(x))
        x = self.relu(self.ff4(x))
        x = self.relu(self.ff5(x))
        x = self.relu(self.ff6(x))
        x = self.relu(self.ff7(x))
        x = self.ff8(x)

        # _, (hn, cn) = self.lstm(x)
        # x = hn.squeeze()
        # x = self.tanh(self.ff1(x))
        # x = self.tanh(self.ff2(x))
        # x = self.tanh(self.ff6(x))
        # x = self.tanh(self.ff7(x))
        # x = self.ff8(x)
        return x

if __name__ == '__main__':
    model = DropPercentModel()
    summary(model, input_size=(10, 20, 5), device='cpu')
    # x = torch.tensor([[10.0, 1.0], [2.0, 1.0]])
    # y = model(x)
