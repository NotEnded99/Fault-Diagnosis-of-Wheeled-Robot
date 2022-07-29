import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Gated_Convolution(nn.Module):
    """
    Gated convolutional layer, each gated convolutional layer contains two conventional convolution operations
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):

        super(Gated_Convolution, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, 1))

    def forward(self, X):

        X = X.permute(0, 3, 1, 2)  # (batch_size, node_features, num_nodes, num_time_steps)
        out = torch.mul((self.conv1(X)), torch.sigmoid(self.conv2(X)))
        
        return out.permute(0, 2, 3, 1)

class STGCM(nn.Module):
    """
    Spatial-temporal graph convolutional module, which consists of a graph convolutional layer,
    two gated convolutional layers, a residual architecture, and a batch normalization layer.
    """
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes, residual=True):
    
        super(STGCM, self).__init__()
        self.temporal1 = Gated_Convolution(in_channels=in_channels, out_channels=out_channels)
        
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        
        self.temporal2 = Gated_Convolution(in_channels=spatial_channels, out_channels=out_channels)
        
        self.batch_norm = nn.BatchNorm2d(num_nodes)
           
        self.reset_parameters()
        
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )
            
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):

        res = self.residual(X.permute(0, 3, 1, 2))
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])  
        t2 = F.relu(torch.matmul(lfs, self.Theta1))  
        t3 = self.temporal2(t2) + res.permute(0, 2, 3, 1)  

        return self.batch_norm(t3)

class STDGCN(nn.Module):

    def __init__(self, num_nodes, node_features, num_class):

        super(STDGCN, self).__init__()
        
        self.stgcm1 = STGCM(in_channels = node_features, out_channels = 32, spatial_channels = 32, num_nodes = num_nodes)
        
        self.stgcm2 = STGCM(in_channels = 32, out_channels = 64, spatial_channels = 64, num_nodes = num_nodes)

        self.last_temporal = Gated_Convolution(in_channels=64, out_channels=16)
        
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        
        self.fully1 = nn.Linear(288, 256)
        self.fully2 = nn.Linear(256, num_class) 
  
        
    def forward(self, data, device, difference):
        
        robot_graph = data["graph"].to(device)[0] 
        A_hat = STDGCN.process_graph(robot_graph)  # Normalize the adjacency matrix

        x_raw = data["sensor_data"].to(device)
        X = STDGCN.difference_layer(x_raw, difference)  # (batch_size, num_nodes, num_time_steps, node_features)
        
        out = self.stgcm1(X, A_hat)
        out = self.stgcm2(out, A_hat)

        out = self.last_temporal(out)  
        out = self.pool(out)  

        out = self.fully1(out.view(out.shape[0], -1))
        out = self.fully2(out)
        
        return out
 
    @staticmethod
    def process_graph(robot_graph):
        degree_matrix = torch.sum(robot_graph, dim=-1, keepdim=False) 
        degree_matrix = degree_matrix.pow(-1)
        degree_matrix[degree_matrix == float("inf")] = 0.
        degree_matrix = torch.diag(degree_matrix) 
        return torch.mm(degree_matrix, robot_graph) 
    
    @staticmethod
    def difference_layer(X, difference): 
        """
        The difference layer, which calculates multi-order backward difference features for feature enhancement
        """
        if  difference == 0:
            out = X
        
        elif difference == 1:
            diff1 = torch.diff(X, n=1, dim=2)
            pad1 = (0, 0, 1, 0)
            diff1 = F.pad(diff1, pad1, mode='constant', value=0)
            out = torch.cat((X, diff1),-1)
            
        elif difference == 2:
            diff1 = torch.diff(X, n=1, dim=2)
            diff2 = torch.diff(diff1, n=1, dim=2)
            pad1 = (0, 0, 1, 0) 
            pad2 = (0, 0, 2, 0)
            diff1 = F.pad(diff1, pad1, mode='constant', value=0)
            diff2 = F.pad(diff2, pad2, mode='constant', value=0)
            out = torch.cat((X, diff1, diff2),-1)

        elif difference == 3:
            diff1 = torch.diff(X, n=1, dim=2)
            diff2 = torch.diff(diff1, n=1, dim=2)
            diff3 = torch.diff(diff2, n=1, dim=2)
            pad1 = (0, 0, 1, 0) 
            pad2 = (0, 0, 2, 0)
            pad3 = (0, 0, 3, 0)
            diff1 = F.pad(diff1, pad1, mode='constant', value=0)
            diff2 = F.pad(diff2, pad2, mode='constant', value=0)
            diff3 = F.pad(diff3, pad3, mode='constant', value=0)
            out = torch.cat((X, diff1, diff2, diff3),-1)     

        elif difference == 4:
            diff1 = torch.diff(X, n=1, dim=2)
            diff2 = torch.diff(diff1, n=1, dim=2)
            diff3 = torch.diff(diff2, n=1, dim=2)
            diff4 = torch.diff(diff3, n=1, dim=2)
            pad1 = (0, 0, 1, 0) 
            pad2 = (0, 0, 2, 0)
            pad3 = (0, 0, 3, 0)
            pad4 = (0, 0, 4, 0)
            diff1 = F.pad(diff1, pad1, mode='constant', value=0)
            diff2 = F.pad(diff2, pad2, mode='constant', value=0)
            diff3 = F.pad(diff3, pad3, mode='constant', value=0)
            diff4 = F.pad(diff4, pad4, mode='constant', value=0)
            out = torch.cat((X, diff1, diff2, diff3, diff4),-1)        
        else:
            raise ValueError("The value of difference should be 0, 1, 2, 3, 4")
            
        return out  # (batch_size, num_nodes, num_time_steps, node_features)
        
    
    
    