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

p = int(sys.argv[1])
f = int(sys.argv[2])
f = 12
trainFile = sys.argv[3]
adjMatrix = sys.argv[4]
graphSplit = sys.argv[5]

adj_mx = pd.read_csv(adjMatrix, index_col=0)
node_mapping = adj_mx.index.tolist()
d = dict(zip(node_mapping, [i for i in range(len(node_mapping))]))
edge_index, edge_weight = dense_to_sparse(torch.tensor(adj_mx.values))
edge_index, _ = remove_self_loops(edge_index)
edge_index, _ = add_self_loops(edge_index, num_nodes=len(node_mapping))

df = pd.read_csv(trainFile, index_col=0)
data_list = []
for index, row in df.iterrows():
    data = [[i] for i in row.tolist()]
    data_list.append(data)

splits = np.load(graphSplit)

# Data is loaded here, test nodes are not loaded
train_node_ids = splits["train_node_ids"]
val_node_ids = splits["val_node_ids"]

train_node_ids = [d[i] for i in train_node_ids]
val_node_ids = [d[i] for i in val_node_ids]

class TrafficDataset2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TrafficDataset2, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load('tensor_part2.pt')

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
        for i in range(p - 1, len(data_list) - f):
            x = np.array(data_list[i - p + 1])
            y = np.array(data_list[i])
            for j in range(i - p + 2, i + 1):
                x = np.dstack((x, data_list[j]))
            
            for j in range(i + 1, i + f):
                y = np.dstack((y, data_list[j]))
            x = torch.Tensor(x)
            y = torch.Tensor(y)
            if x.ndim == 2:
                x = torch.unsqueeze(x, 2)
            if y.ndim == 2:
                y = torch.unsqueeze(y, 2)
            data = Data(x=torch.Tensor(x), y=torch.Tensor(y.squeeze(1)))
            final_data.append(data)
        data, slices = self.collate(final_data)
        torch.save((data, slices), 'tensor_part2.pt')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = A3TGCN(in_channels=1, out_channels=4, periods=p) 
        self.linear = torch.nn.Linear(4, 12)
  
    def forward(self, data): 
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        output = self.linear(h)
        return output
        
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
dataset = TrafficDataset2(root='')
train_loader = DataLoader(dataset,batch_size=batch_size)
test_loader = DataLoader(dataset)
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
crit = torch.nn.L1Loss()
model.train()

for epoch in range(10):
    loss = train(train_loader, train_node_ids) # Passing train nodes to train() function to compute loss only on train nodes
    # train_acc = evaluate(test_loader, train_node_ids)
    # val_acc = evaluate(test_loader, val_node_ids)
    print(epoch, loss)

torch.save(model.state_dict(), 'cs5190471_task2.model')

f = open("adjPath.txt", "w")
f.write(adjMatrix)
f.close()