import torch
import torch.nn as nn

# Original ConvLSTM cell as proposed by Shi et al.
class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       padding,
                       frame_size,
                       batch_size,
                       device,
                       activation="tanh"):

        super(ConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        self.Wxi = nn.Conv2d(
            in_channels=in_channels, #we're doing conv of X and H_{k-1} together!
            out_channels=out_channels, # why 4?! read above!
            kernel_size=kernel_size,
            padding=padding)
        self.Wxf = nn.Conv2d(
            in_channels=in_channels, #we're doing conv of X and H_{k-1} together!
            out_channels=out_channels, # why 4?! read above!
            kernel_size=kernel_size,
            padding=padding)
        self.Wxo = nn.Conv2d(
            in_channels=in_channels, #we're doing conv of X and H_{k-1} together!
            out_channels=out_channels, # why 4?! read above!
            kernel_size=kernel_size,
            padding=padding)
        self.Wxc = nn.Conv2d(
            in_channels=in_channels, #we're doing conv of X and H_{k-1} together!
            out_channels=out_channels, # why 4?! read above!
            kernel_size=kernel_size,
            padding=padding)

        self.Whi = nn.Conv2d(
            in_channels=in_channels, #we're doing conv of X and H_{k-1} together!
            out_channels=out_channels, # why 4?! read above!
            kernel_size=kernel_size,
            padding=padding)
        self.Whf = nn.Conv2d(
            in_channels=in_channels, #we're doing conv of X and H_{k-1} together!
            out_channels=out_channels, # why 4?! read above!
            kernel_size=kernel_size,
            padding=padding)
        self.Who = nn.Conv2d(
            in_channels=in_channels, #we're doing conv of X and H_{k-1} together!
            out_channels=out_channels, # why 4?! read above!
            kernel_size=kernel_size,
            padding=padding)
        self.Whc = nn.Conv2d(
            in_channels=in_channels, #we're doing conv of X and H_{k-1} together!
            out_channels=out_channels, # why 4?! read above!
            kernel_size=kernel_size,
            padding=padding)

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

        self.bi = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.bf = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.bc = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.bo = nn.Parameter(torch.Tensor(out_channels, *frame_size))

        # FALTA INICIALIZAR OS OUTROS parametros
        # Initialize the parameters
        nn.init.xavier_uniform_(self.Wxi.weight)
        nn.init.xavier_uniform_(self.Wxf.weight)
        nn.init.xavier_uniform_(self.Wxo.weight)
        nn.init.xavier_uniform_(self.Wxc.weight)
        nn.init.xavier_uniform_(self.Whi.weight)
        nn.init.xavier_uniform_(self.Whf.weight)
        nn.init.xavier_uniform_(self.Who.weight)
        nn.init.xavier_uniform_(self.Whc.weight)
        nn.init.constant_(self.W_ci, 0)
        nn.init.constant_(self.W_cf, 0)
        nn.init.constant_(self.W_co, 0)
        nn.init.constant_(self.bi, 0)
        nn.init.constant_(self.bf, 0)
        nn.init.constant_(self.bc, 0)
        nn.init.constant_(self.bo, 0)

        # Initialize Cell Input
        self.C = torch.zeros(batch_size, out_channels,
        frame_size[0], frame_size[1], device=device)


    def forward(self, X, H_prev):

        # input gate
        inputGate = torch.sigmoid(self.Wxi(X) + self.Whi(H_prev) + self.W_ci*self.C + self.bi)
        # forget gate
        forgetGate = torch.sigmoid(self.Wxf(X) + self.Whf(H_prev)+ self.W_cf*self.C + self.bf)
        # C_t
        self.C = forgetGate*self.C + inputGate * self.activation(self.Wxc(X) + self.Whc(H_prev) + self.bc)
        # output state
        out = torch.sigmoid(self.Wxo(X) + self.Who(H_prev) + self.W_co*self.C + self.bo)
        # hidden state
        H = out * self.activation(self.C)

        return out, H


class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels,
    kernel_size, padding, activation, frame_size, device):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels
        self.device = device
        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels,
        kernel_size, padding, frame_size, activation)

    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        #print("input size: ", X.size())
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len,
        height, width, device=self.device)

        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels,
        height, width, device=self.device)

        # Initialize Cell Input
        C = torch.zeros(batch_size,self.out_channels,
        height, width, device=self.device)

        # Unroll over time steps
        for time_step in range(seq_len):

            out, H, C = self.convLSTMcell(X[:,:,time_step], H, C)

            output[:,:,time_step] = H

        return out, output

class Seq2Seq(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding,
    activation, frame_size, num_layers):

        super(Seq2Seq, self).__init__()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding,
                activation=activation, frame_size=frame_size)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        )

        # Add rest of the layers
        for l in range(2, num_layers+1):
            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding,
                    activation=activation, frame_size=frame_size)
                )
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
                )

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):

        # Forward propagation through all the layers
        output = self.sequential(X)

        # Return only the last output frame
        output = self.conv(output[:,:,-1])

        return nn.Sigmoid()(output)

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

def temp():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Seq2Seq(num_channels=1,
                    num_kernels=64,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    activation="relu",
                    frame_size=(64, 64),
                    num_layers=3).to(device)

    optim = Adam(model.parameters(), lr=1e-4)

    # Binary Cross Entropy, target pixel values either 0 or 1
    criterion = nn.BCELoss(reduction='sum')


class ConvLSTMCell2(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell2, self).__init__()
        self.hidden_channels = hidden_channels

        # Convolutional layers for input-to-hidden, hidden-to-hidden, and cell-to-input transformations
        self.Wxi = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.Wci = nn.Parameter(torch.Tensor(hidden_channels, 1, 1))  # Hadamard parameter
        self.bi = nn.Parameter(torch.Tensor(hidden_channels, 1, 1))

        # Initialize the parameters
        nn.init.xavier_uniform_(self.Wxi.weight)
        nn.init.xavier_uniform_(self.Whi.weight)
        nn.init.constant_(self.Wci, 0)
        nn.init.constant_(self.bi, 0)

    def forward(self, x_t, H_t_1, c_t_1):
        # Convolution operations
        Wxi_x_t = self.Wxi(x_t)
        Whi_H_t_1 = self.Whi(H_t_1)

        # Element-wise operations
        Wci_c_t_1 = self.Wci * c_t_1
        sum_all = Wxi_x_t + Whi_H_t_1 + Wci_c_t_1 + self.bi

        # Apply the sigmoid activation function
        i_t = torch.sigmoid(sum_all)

        return i_t
'''
# Example usage
batch_size, input_channels, height, width = 4, 3, 64, 64
hidden_channels = 16
kernel_size = 3

# Create a ConvLSTMCell
conv_lstm_cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)

# Generate some random data
x_t = torch.randn(batch_size, input_channels, height, width)
H_t_1 = torch.randn(batch_size, hidden_channels, height, width)
c_t_1 = torch.randn(batch_size, hidden_channels, height, width)

# Forward pass
i_t = conv_lstm_cell(x_t, H_t_1, c_t_1)
print(i_t.shape)  # Should output: torch.Size([4, 16, 64, 64])'''