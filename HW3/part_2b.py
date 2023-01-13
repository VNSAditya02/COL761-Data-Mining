import pandas as pd
import numpy as np
import sys
import torch
from torch_geometric.data import Data
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, dense_to_sparse
from torch_geometric.nn import TopKPooling, GATConv, SAGEConv, GCNConv
from torch_geometric_temporal.nn.recurrent import GConvLSTM, A3TGCN
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from numpy import savez_compressed

p = int(sys.argv[1])
f = int(sys.argv[2])
testFile = sys.argv[3]
outputFile = sys.argv[4]
modelFile = sys.argv[5]

file = open("adjPath.txt", "r")
adjMatrix = file.readline().strip('\n')
file.close()

adj_mx = pd.read_csv(adjMatrix, index_col=0)
node_mapping = adj_mx.index.tolist()
d = dict(zip(node_mapping, [i for i in range(len(node_mapping))]))
edge_index, edge_weight = dense_to_sparse(torch.tensor(adj_mx.values))
edge_index, _ = remove_self_loops(edge_index)
edge_index, _ = add_self_loops(edge_index, num_nodes=len(node_mapping))

test_data = np.load(testFile)
data_list = np.array(test_data['x'])
s = test_data['x'].shape
n_windows = s[0]

# df = pd.read_csv('d1_X.csv', index_col=0)[:100]
# data_list = []
# for index, row in df.iterrows():
#     data = [[i] for i in row.tolist()]
#     data_list.append(data)

class TrafficDataset2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TrafficDataset2, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load('tensor_part2_test.pt')

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass
    
    def process(self):
        final_data = []
        for i in range(n_windows):
            
            x = torch.Tensor(np.transpose(data_list[i], axes = (1,0)))
            x = torch.unsqueeze(x, 1)
            data = Data(x=x)
            final_data.append(data)
        data, slices = self.collate(final_data)
        torch.save((data, slices), 'tensor_part2_test.pt')

        # final_data = []
        # for i in range(p - 1, len(data_list) - f):
        #     x = np.array(data_list[i - p + 1])
        #     y = np.array(data_list[i])
        #     for j in range(i - p + 2, i + 1):
        #         x = np.dstack((x, data_list[j]))
            
        #     for j in range(i + 1, i + f):
        #         y = np.dstack((y, data_list[j]))
        #     x = torch.Tensor(x)
        #     y = torch.Tensor(y)
        #     if x.ndim == 2:
        #         x = torch.unsqueeze(x, 2)
        #     if y.ndim == 2:
        #         y = torch.unsqueeze(y, 2)
        #     data = Data(x=torch.Tensor(x), y=torch.Tensor(y.squeeze(1)))
        #     final_data.append(data)
        # data, slices = self.collate(final_data)
        # torch.save((data, slices), 'tensor_part2_test.pt')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = A3TGCN(in_channels=1, out_channels=4, periods=p) 
        self.linear = torch.nn.Linear(4, 12)
  
    def forward(self, data): 
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        output = self.linear(h)
        return output[:,:f]

def evaluate(loader):
    model.eval()
    with torch.no_grad():
        predictions = []
        # actual = []
        for data in loader:
            data.edge_index = edge_index
            data = data.to(device)
            pred = model(data).detach().cpu()
            # label = data.y.detach().cpu()
            predictions.append(pred.numpy())
            # actual.append(label.numpy())
    # print(crit(torch.Tensor(predictions), torch.Tensor(actual)))
    predictions = np.transpose(predictions, axes = (0, 2, 1))
    predictions = torch.Tensor(predictions)
    savez_compressed(outputFile, y=predictions)

batch_size= 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TrafficDataset2(root='')
test_loader = DataLoader(dataset)

model = Net()
model.load_state_dict(torch.load(modelFile, map_location=torch.device('cpu')))
model.to(device)

crit = torch.nn.L1Loss()
evaluate(test_loader)