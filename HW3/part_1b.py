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
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from datetime import datetime

testFile = sys.argv[1]
outputFile = sys.argv[2]
modelFile = sys.argv[3]

f = open("adjPath.txt", "r")
adjMatrix = f.readline().strip('\n')
f.close()

adj_mx = pd.read_csv(adjMatrix, index_col=0)
node_mapping = adj_mx.index.tolist()
d = dict(zip(node_mapping, [i for i in range(len(node_mapping))]))
edge_index, edge_weight = dense_to_sparse(torch.tensor(adj_mx.values))
edge_index, _ = remove_self_loops(edge_index)
edge_index, _ = add_self_loops(edge_index, num_nodes=len(node_mapping))

df = pd.read_csv(testFile, index_col=0)

class TrafficDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TrafficDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load('tensor_test.pt')

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass
    
    def process(self):
        data_list = []
        for index, row in df.iterrows():
            x = [[i] for i in row.tolist()]
            x = torch.Tensor(x)
            data = Data(x=x)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), 'tensor_test.pt')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(1, 4)
        self.conv2 = SAGEConv(4, 1) 
  
    def forward(self, data): 
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x.float(), edge_index)
        return x

def evaluate(loader):
    model.eval()
    with torch.no_grad():
        output = []
        predictions = []
        for data in loader:
            data.edge_index = edge_index
            data = data.to(device)
            pred = model(data).detach().cpu()
            predictions.extend(pred.numpy())
            output.append(pred.numpy())
        output = np.array(output).squeeze(2)
        output_df = pd.DataFrame(output, columns= df.columns)
        times = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in df.index]
        delta = times[1] - times[0]
        times = [i + delta for i in times]
        output_df.index = times
        output_df.to_csv(outputFile)
        return 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TrafficDataset(root='')
test_loader = DataLoader(dataset)

model = Net()
model.load_state_dict(torch.load(modelFile, map_location=torch.device('cpu')))
model.to(device)

evaluate(test_loader)