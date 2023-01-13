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

trainFile = sys.argv[1]
adjMatrix = sys.argv[2]
graphSplit = sys.argv[3]

adj_mx = pd.read_csv(adjMatrix, index_col=0)
node_mapping = adj_mx.index.tolist()
d = dict(zip(node_mapping, [i for i in range(len(node_mapping))]))
edge_index, edge_weight = dense_to_sparse(torch.tensor(adj_mx.values))
edge_index, _ = remove_self_loops(edge_index)
edge_index, _ = add_self_loops(edge_index, num_nodes=len(node_mapping))

df = pd.read_csv(trainFile, index_col=0)

splits = np.load(graphSplit)

# Data is loaded here, test nodes are not loaded
train_node_ids = splits["train_node_ids"]
val_node_ids = splits["val_node_ids"]

train_node_ids = [d[i] for i in train_node_ids]
val_node_ids = [d[i] for i in val_node_ids]

class TrafficDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TrafficDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load('tensor.pt')

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
        x = []
        i = 0
        for index, row in df.iterrows():
            if i != 0:
                y = [[i] for i in row.tolist()]
                y = torch.Tensor(y)
                data = Data(x=x, y=y)
                data_list.append(data)
                x = y
            else:
                x = [[i] for i in row.tolist()]
                x = torch.Tensor(x)
            i += 1
        data, slices = self.collate(data_list)
        torch.save((data, slices), 'tensor.pt')

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
        
def train(loader, nodes):
    model.train()
    loss_all = 0
    for data in loader:
        data.edge_index = edge_index
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = crit(output[nodes], label[nodes]) # Loss is computed here over train nodes only
        loss.backward()
        loss_all += loss.item()*data.num_graphs
        optimizer.step()
    return loss_all

def evaluate(loader, nodes):
    model.eval()
    with torch.no_grad():
        loss_all = 0
        predictions = []
        actual = []
        for data in loader:
            data.edge_index = edge_index
            data = data.to(device)
            pred = model(data).detach().cpu()[nodes]
            label = data.y.detach().cpu()[nodes]
            predictions.extend(pred.numpy())
            actual.extend(label.numpy())
        predictions = np.concatenate(predictions, axis=0)
        actual = np.concatenate(actual, axis=0)
        return crit(torch.Tensor(predictions), torch.Tensor(actual))

batch_size= 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dataset = TrafficDataset(root='')
train_loader = DataLoader(dataset, batch_size=batch_size)
test_loader = DataLoader(dataset)
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
crit = torch.nn.L1Loss()
model.train()
for epoch in range(10):
    loss = train(train_loader, train_node_ids) # Passing train nodes to train() function to compute loss only on train nodes
    # train_acc = evaluate(test_loader, train_node_ids)
    # val_acc = evaluate(test_loader, val_node_ids)
    print(epoch, loss)

torch.save(model.state_dict(), 'cs5190471_task1.model')

f = open("adjPath.txt", "w")
f.write(adjMatrix)
f.close()
