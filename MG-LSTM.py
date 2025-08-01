import pandas as pd
import networkx as nx
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

class MGLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=1):
        super(MGLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        # GAT layers for transforming input and hidden states
        self.gat_input = GATConv(input_dim, hidden_dim, heads=num_heads)
        self.gat_hidden = GATConv(hidden_dim * num_heads, hidden_dim, heads=1)
        
        # Forget gate components
        self.gat_forget_x = GATConv(input_dim, hidden_dim, heads=1)
        self.gat_forget_h = GATConv(hidden_dim, hidden_dim, heads=1)
        
        # Input gate components
        self.gat_input_x = GATConv(input_dim, hidden_dim, heads=1)
        self.gat_input_h = GATConv(hidden_dim, hidden_dim, heads=1)
        
        # Output gate components
        self.gat_output_x = GATConv(input_dim, hidden_dim, heads=1)
        self.gat_output_h = GATConv(hidden_dim, hidden_dim, heads=1)
        
        # Candidate cell state components
        self.gat_candidate_x = GATConv(input_dim, hidden_dim, heads=1)
        self.gat_candidate_h = GATConv(hidden_dim, hidden_dim, heads=1)
        
        # Missing information prediction components
        self.W_gamma1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_gamma2 = nn.Linear(hidden_dim, hidden_dim)
        self.W_beta1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_beta2 = nn.Linear(hidden_dim, hidden_dim)

    def predict_missing_info(self, h_v, h_N_v):
        gamma = torch.tanh(self.W_gamma1(h_v) + self.W_gamma2(h_N_v))
        beta = torch.tanh(self.W_beta1(h_v) + self.W_beta2(h_N_v))
        r = torch.zeros_like(h_v)
        r_v = (gamma + 1) * r + beta
        m_v = h_v + r_v - h_N_v
        return m_v

    def forward(self, x, edge_index, h_c, batch=None):
        # Apply GAT to input
        x_transformed = self.gat_input(x, edge_index)
        
        if h_c is None:
            h, c = (torch.zeros(x.size(0), self.hidden_dim, device=x.device),
                    torch.zeros(x.size(0), self.hidden_dim, device=x.device))
        else:
            h, c = h_c

        h_N = self.gat_hidden(h, edge_index)
        m = self.predict_missing_info(h, h_N)
        h_N = h_N + m

        # Forget gate
        f = torch.sigmoid(self.gat_forget_x(x, edge_index) + 
                          self.gat_forget_h(h_N, edge_index))

        # Input gate
        i = torch.sigmoid(self.gat_input_x(x, edge_index) + 
                          self.gat_input_h(h_N, edge_index))

        # Candidate cell state
        c_tilde = torch.tanh(self.gat_candidate_x(x, edge_index) + 
                             self.gat_candidate_h(h_N, edge_index))

        # Update cell state
        c_new = f * c + i * c_tilde

        # Output gate
        o = torch.sigmoid(self.gat_output_x(x, edge_index) + 
                          self.gat_output_h(h_N, edge_index))

        # New hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class EdgeClassifier(nn.Module):
    def __init__(self, node_embedding_dim):
        super(EdgeClassifier, self).__init__()
        edge_embedding_dim = 2 * node_embedding_dim
        # Logistic regression as a single linear layer
        self.linear = nn.Linear(edge_embedding_dim, 1)
    
    def forward(self, edge_embeddings):
        # Apply logistic regression and use sigmoid activation
        logits = self.linear(edge_embeddings)
        return torch.sigmoid(logits)

class EnhancedTemporalGraphNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(EnhancedTemporalGraphNetwork, self).__init__()
        self.mglstm = MGLSTM(input_dim, hidden_dim)
        self.num_layers = num_layers
        self.edge_classifier = EdgeClassifier(hidden_dim)
        
    def create_edge_embeddings(self, node_embeddings, edge_index):
        src_embeddings = node_embeddings[edge_index[0]]
        dst_embeddings = node_embeddings[edge_index[1]]
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        return edge_embeddings
    
    def forward(self, x, edge_index, batch=None):
        h_c = None
        for _ in range(self.num_layers):
            h, c = self.mglstm(x, edge_index, h_c, batch)
            h_c = (h, c)

        edge_embeddings = self.create_edge_embeddings(h, edge_index)
        edge_predictions = self.edge_classifier(edge_embeddings)
        
        return edge_predictions, h
        

# The rest of your code remains unchanged...

def create_edge_labels(G, labels, edge_index, node_to_idx):
    edge_labels = []
    for i in range(edge_index.size(1)):
        src_idx = edge_index[0][i].item()
        dst_idx = edge_index[1][i].item()
        
        # Convert indices back to original node IDs
        src_node = list(G.nodes())[src_idx]
        dst_node = list(G.nodes())[dst_idx]
        
        # Create edge label based on source and destination node labels
        src_label = 1 if labels[src_node][0] == 'collection_abnormal' else 0
        dst_label = 1 if labels[dst_node][0] == 'collection_abnormal' else 0
        
        # Edge is labeled as abnormal if either source or destination is abnormal
        edge_labels.append(float(src_label or dst_label))
    
    return torch.tensor(edge_labels, dtype=torch.float)

def weighted_cross_entropy_loss(predictions, targets, pos_weight):
    """
    Custom weighted cross entropy loss
    N1: number of positive samples
    N2: number of negative samples
    """
    epsilon = 1e-7  # Small constant to prevent log(0)
    predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
    
    # Calculate weighted loss
    loss = -(pos_weight * targets * torch.log(predictions) + 
             (1 - targets) * torch.log(1 - predictions))
    
    return loss.mean()

def create_graph(data):
    G = nx.DiGraph()
    nodes = set(data['from_address'].tolist() + data['to_address'].tolist())
    G.add_nodes_from(nodes)
    for _, row in data.iterrows():
        G.add_edge(row['from_address'], row['to_address'], weight=row['timestamp'])
    return G

def calculate_fraud_and_antifraud_scores(G):
    fraud_scores = nx.out_degree_centrality(G)
    antifraud_scores = nx.in_degree_centrality(G)
    return fraud_scores, antifraud_scores

def label_nodes(fraud_scores, antifraud_scores, fraud_threshold=0.01, antifraud_threshold=0.01):
    labels = {}
    for node in fraud_scores:
        collection_label = 'collection_abnormal' if fraud_scores[node] > fraud_threshold else 'collection_normal'
        pay_label = 'pay_normal' if antifraud_scores[node] > antifraud_threshold else 'pay_abnormal'
        labels[node] = (collection_label, pay_label)
    return labels

def label_edges(G):
    ego_networks = defaultdict(nx.DiGraph)
    for u, v, data in G.edges(data=True):
        ego_networks[u].add_edge(u, v, weight=data['weight'])
        ego_networks[v].add_edge(u, v, weight=data['weight'])
    return ego_networks

def count_edges(ego_network, label):
    count = 0
    for u, v, data in ego_network.edges(data=True):
        if label in data:
            count += 1
    return count

def common_eval(ego_networks):
    neighbors = {}
    for node, ego_net in ego_networks.items():
        neighbors[node] = list(ego_net.neighbors(node))
    return neighbors

def extract_features(G, node):
    ego_networks = label_edges(G)
    neighbors = common_eval(ego_networks)
    T1 = count_edges(ego_networks[node], label='collection_normal')
    T2 = count_edges(ego_networks[node], label='collection_abnormal')
    T3 = count_edges(ego_networks[node], label='payment_normal')
    T4 = count_edges(ego_networks[node], label='payment_abnormal')
    
    node_features = [T1, T2, len(neighbors.get(node, [])), 
                    T3, T4, len(neighbors.get(node, [])), 
                    G.in_degree(node), G.out_degree(node)]
    return node_features

def create_data_list(G_list, labels_list):
    data_list = []
    for G, labels in zip(G_list, labels_list):
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes)}
        
        # Create edge index for the entire graph
        edge_index = torch.tensor([(node_to_idx[u], node_to_idx[v]) 
                                  for u, v in G.edges], dtype=torch.long).t().contiguous()
        
        # Create feature matrix for all nodes
        x = torch.tensor([extract_features(G, node) for node in G.nodes], 
                        dtype=torch.float)
        
        # Create edge labels
        edge_labels = create_edge_labels(G, labels, edge_index, node_to_idx)
        
        # Create a single Data object for the entire graph
        data = Data(x=x, edge_index=edge_index, y=edge_labels)
        data_list.append(data)
    
    return data_list

def train_model(model, train_loader, epochs=100, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(epochs):
        for data in train_loader:
            optimizer.zero_grad()
            edge_predictions, _ = model(data.x, data.edge_index, data.batch)
            loss = weighted_cross_entropy_loss(edge_predictions, data.y, pos_weight=1.0)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model

def evaluate_model(model, loader):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            edge_predictions, _ = model(batch.x, batch.edge_index, batch.batch)
            all_predictions.append(edge_predictions.squeeze().cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    auc_score = roc_auc_score(all_labels, all_predictions)
    precision = precision_score(all_labels, [1 if p > 0.5 else 0 for p in all_predictions])
    recall = recall_score(all_labels, [1 if p > 0.5 else 0 for p in all_predictions])
    f1 = f1_score(all_labels, [1 if p > 0.5 else 0 for p in all_predictions])
    return auc_score, precision, recall, f1


def main():
    
    # Load the dataset
    file_path = 'soc-sign-bitcoinalpha.csv'
    data = pd.read_csv(file_path)

    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

    # Sort data by timestamp
    data = data.sort_values(by='timestamp')

    # Split data into 31-day time slices
    data['time_slice'] = (data['timestamp'] - data['timestamp'].min()).dt.days // 31

    # Combine the last few sparse time slices into a single time slice
    threshold = 5  # Combine slices with fewer than 20 entries
    combined_time_slice = max(data['time_slice']) - 1
    data.loc[data['time_slice'] >= combined_time_slice, 'time_slice'] = combined_time_slice

    # Verify the new distribution of entries across time slices
    new_time_slice_counts = data['time_slice'].value_counts().sort_index()

    # Display the new distribution
    print(new_time_slice_counts)

    
    # Create a list of graphs and labels for each time slice
    G_list = []
    labels_list = []
    
    for time_slice in data['time_slice'].unique():
        slice_data = data[data['time_slice'] == time_slice]
        G = create_graph(slice_data)
        fraud_scores, antifraud_scores = calculate_fraud_and_antifraud_scores(G)
        labels = label_nodes(fraud_scores, antifraud_scores)
        G_list.append(G)
        labels_list.append(labels)
    
    # Check if we have enough data slices for splitting
    if len(G_list) > 2:
        # Split data into train, validation, and test sets (60%, 20%, 20%)
        train_G, temp_G, train_labels, temp_labels = train_test_split(G_list, labels_list, test_size=0.4, shuffle=False)
        val_G, test_G, val_labels, test_labels = train_test_split(temp_G, temp_labels, test_size=0.5, shuffle=False)
    else:
        # If there's not enough time slices, use all data for training and skip validation/testing
        train_G, train_labels = G_list, labels_list
        val_G, val_labels, test_G, test_labels = [], [], [], []
    
    # Create dataset and dataloader
    train_data_list = create_data_list(train_G, train_labels)
    val_data_list = create_data_list(val_G, val_labels) if val_G else []
    test_data_list = create_data_list(test_G, test_labels) if test_G else []
    
    train_loader = DataLoader(train_data_list, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=64, shuffle=False) if val_data_list else None
    test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False) if test_data_list else None
    
    # Initialize and train model
    model = EnhancedTemporalGraphNetwork(
        input_dim=8,
        hidden_dim=16,
        num_layers=2
    )
    
    # Train the model
    trained_model = train_model(model, train_loader)
    
    # Evaluate the model on the test set if it exists
    if test_loader:
        test_auc, precision, recall, f1 = evaluate_model(trained_model, test_loader)
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1:.4f}")
    else:
        print("Not enough data to create a test set.")
    
    return trained_model

if __name__ == "__main__":
    trained_model = main()
