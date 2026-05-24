import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any

def generate_transaction_graph(
    num_users: int = 800,
    num_merchants: int = 200,
    num_normal_transactions: int = 5000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generates a realistic, synthetic transaction graph containing user accounts,
    merchant accounts, transaction edges, and injected structural fraud patterns.
    
    Returns a dictionary containing:
        - node_features (torch.Tensor): Feature matrix of shape [num_nodes, 5]
        - node_types (torch.Tensor): Binary indicator [num_nodes] where 0 = User, 1 = Merchant
        - edge_index (torch.Tensor): Graph edge matrix of shape [2, num_edges]
        - edge_features (torch.Tensor): Transaction feature matrix of shape [num_edges, 5]
        - edge_labels (torch.Tensor): Fraud labels of shape [num_edges] (0=Normal, 1=Fraud)
        - df_transactions (pd.DataFrame): Tabular pandas dataframe of transactions for SHAP & Streamlit
        - df_nodes (pd.DataFrame): Tabular pandas dataframe of nodes for analytics
        - nx_graph (nx.DiGraph): NetworkX graph object for graph analytics
    """
    np.random.seed(seed)
    num_nodes = num_users + num_merchants
    
    # ----------------------------------------------------
    # 1. NODE GENERATION & FEATURE ENGINEERING
    # ----------------------------------------------------
    # Node features:
    # 0: log10(Balance) or Credit Limit
    # 1: Account Age in Days (scaled)
    # 2: Location Risk Score (0.0 to 1.0)
    # 3: Device Trust Score (0.0 to 1.0)
    # 4: Behavioral Risk Multiplier (0.0 to 1.0)
    
    node_features = np.zeros((num_nodes, 5), dtype=np.float32)
    node_types = np.zeros(num_nodes, dtype=np.int64) # 0 = User, 1 = Merchant
    
    # Generate Users
    for i in range(num_users):
        balance = np.random.exponential(scale=3000.0) + 10.0
        age = np.random.randint(10, 365 * 5)
        loc_risk = np.random.beta(a=1, b=5) # skewed towards low risk
        dev_trust = np.random.beta(a=5, b=2) # skewed towards high trust
        behav_risk = np.random.beta(a=1, b=8) # skewed towards low risk
        
        node_features[i] = [
            np.log10(balance),
            age / 1825.0, # scale to [0, 1] range
            loc_risk,
            dev_trust,
            behav_risk
        ]
        node_types[i] = 0
        
    # Generate Merchants
    for i in range(num_users, num_nodes):
        balance = np.random.exponential(scale=100000.0) + 1000.0
        age = np.random.randint(90, 365 * 10)
        loc_risk = np.random.beta(a=1, b=6) # merchants are generally fixed
        dev_trust = 0.95 # merchants have highly trusted setups
        # Feature 4 for merchants represents sector risk (e.g. crypto, high-end retail, grocery)
        sector_risk = np.random.choice([0.1, 0.2, 0.4, 0.8], p=[0.5, 0.3, 0.1, 0.1])
        
        node_features[i] = [
            np.log10(balance),
            age / 3650.0, # scale to [0, 1]
            loc_risk,
            dev_trust,
            sector_risk
        ]
        node_types[i] = 1

    # ----------------------------------------------------
    # 2. BASE TRANSACTION GENERATION (NORMAL BEHAVIOR)
    # ----------------------------------------------------
    # Edge features:
    # 0: Transaction Amount (scaled log10)
    # 1: Hour of Day (0.0 to 1.0, cyclical scaled)
    # 2: Day of Week (0.0 to 1.0, scaled)
    # 3: Geographic Distance (0.0 to 1.0, scaled)
    # 4: Online Transaction Flag (0.0 or 1.0)
    
    src_nodes = []
    dst_nodes = []
    edge_feats = []
    edge_labels = []
    timestamps = []
    
    # Start time
    base_time = pd.Timestamp("2026-05-22 00:00:00")
    
    # Generate normal transactions (primarily users paying merchants, occasionally user-to-user transfers)
    for _ in range(num_normal_transactions):
        # Pick a user (normal accounts have more activity)
        u_idx = np.random.randint(0, num_users)
        
        # 90% of the time transactions are User -> Merchant, 10% User -> User
        if np.random.rand() < 0.90:
            m_idx = np.random.randint(num_users, num_nodes)
        else:
            m_idx = np.random.randint(0, num_users)
            while m_idx == u_idx:
                m_idx = np.random.randint(0, num_users)
                
        # Normal transaction amounts
        amount = np.random.exponential(scale=80.0) + 1.5
        hour = np.random.choice(np.arange(24), p=[
            0.01, 0.01, 0.005, 0.005, 0.01, 0.02, 
            0.04, 0.06, 0.07, 0.07, 0.06, 0.06,
            0.07, 0.06, 0.05, 0.06, 0.07, 0.07,
            0.08, 0.05, 0.03, 0.02, 0.01, 0.01
        ]) # realistic hourly distribution (sums to exactly 1.0)
        day_of_week = np.random.randint(0, 7)
        dist = np.random.exponential(scale=15.0) # normal distance (km)
        online = 1.0 if np.random.rand() < 0.65 else 0.0
        
        # Timestamps spread over a 7-day period
        days_offset = np.random.randint(0, 7)
        ts = base_time + pd.Timedelta(days=days_offset, hours=hour, minutes=np.random.randint(0, 60))
        
        src_nodes.append(u_idx)
        dst_nodes.append(m_idx)
        edge_feats.append([
            np.log10(amount),
            hour / 23.0,
            day_of_week / 6.0,
            min(dist / 500.0, 1.0), # scale
            online
        ])
        edge_labels.append(0)
        timestamps.append(ts)
        
    # ----------------------------------------------------
    # 3. INJECT STRUCTURAL FRAUD PATTERNS (ANOMALIES)
    # ----------------------------------------------------
    # We inject three specific types of fraudulent behaviors:
    
    # Pattern A: Carding / Velocity Attack (Credit Card Theft)
    # A single compromised user account executes rapid-fire, high-amount transactions
    # to multiple random merchants in a short time frame, usually late at night, with high geo-distance.
    num_velocity_attacks = 15
    for _ in range(num_velocity_attacks):
        victim_idx = np.random.randint(0, num_users)
        # Select 5 to 8 merchants to target in rapid fire
        num_targets = np.random.randint(5, 9)
        target_merchants = np.random.choice(np.arange(num_users, num_nodes), size=num_targets, replace=False)
        
        # Fraudulent attributes
        hour = np.random.randint(0, 4) # odd hours (midnight to 4 AM)
        day_of_week = np.random.randint(0, 7)
        online = 1.0 # online velocity carding
        days_offset = np.random.randint(0, 7)
        
        start_minute = np.random.randint(0, 30)
        for step, m_idx in enumerate(target_merchants):
            amount = np.random.uniform(500.0, 2000.0) # large amounts
            dist = np.random.uniform(150.0, 2000.0) # highly suspicious geographic discrepancy
            ts = base_time + pd.Timedelta(days=days_offset, hours=hour, minutes=start_minute + step * 2) # every 2 minutes!
            
            src_nodes.append(victim_idx)
            dst_nodes.append(m_idx)
            edge_feats.append([
                np.log10(amount),
                hour / 23.0,
                day_of_week / 6.0,
                min(dist / 500.0, 1.0),
                online
            ])
            edge_labels.append(1) # Fraud
            timestamps.append(ts)
            
    # Pattern B: Money Laundering Mule Network (Star Topology aggregation)
    # Multiple low-risk ordinary user accounts (mules) suddenly transfer funds to a single merchant
    # or specific account node (the aggregator), which then disperses it.
    num_mule_networks = 5
    for _ in range(num_mule_networks):
        aggregator_idx = np.random.randint(0, num_nodes) # could be user or merchant
        # Select 12 to 20 unique "mules"
        num_mules = np.random.randint(12, 21)
        mule_indices = np.random.choice(np.arange(0, num_users), size=num_mules, replace=False)
        mule_indices = [m for m in mule_indices if m != aggregator_idx]
        
        hour = np.random.randint(10, 18) # standard daytime hours to avoid heuristic flags
        day_of_week = np.random.randint(0, 7)
        days_offset = np.random.randint(0, 7)
        online = 1.0
        
        # Mules transfer highly structured "sub-limit" amounts (e.g. just below reporting limits)
        for step, mule_idx in enumerate(mule_indices):
            amount = np.random.uniform(9500.0, 9999.0) # structure below $10,000 threshold
            dist = np.random.uniform(5.0, 300.0)
            # Timestamps are tightly clustered (e.g., within a 1-hour window)
            ts = base_time + pd.Timedelta(days=days_offset, hours=hour, minutes=np.random.randint(0, 60))
            
            src_nodes.append(mule_idx)
            dst_nodes.append(aggregator_idx)
            edge_feats.append([
                np.log10(amount),
                hour / 23.0,
                day_of_week / 6.0,
                min(dist / 500.0, 1.0),
                online
            ])
            edge_labels.append(1) # Laundering is Fraud
            timestamps.append(ts)

    # Pattern C: Risky Bipartite Spikes (High-Degree Merchant)
    # A newly created, high-risk merchant suddenly has a massive spikes of inbound payments 
    # from a variety of risky users.
    num_bipartite_spikes = 4
    for _ in range(num_bipartite_spikes):
        # Pick a high-risk merchant node
        risk_merchants = [i for i in range(num_users, num_nodes) if node_features[i, 4] >= 0.4]
        if len(risk_merchants) == 0:
            merchant_idx = np.random.randint(num_users, num_nodes)
        else:
            merchant_idx = np.random.choice(risk_merchants)
            
        # Select 15 to 25 accounts that are moderately higher risk
        num_risky_users = np.random.randint(15, 26)
        risky_users = np.random.choice(np.arange(0, num_users), size=num_risky_users, replace=False)
        
        hour = np.random.randint(1, 6) # odd morning hours
        day_of_week = np.random.randint(0, 7)
        days_offset = np.random.randint(0, 7)
        online = 1.0
        
        for step, u_idx in enumerate(risky_users):
            amount = np.random.exponential(scale=500.0) + 100.0 # large amounts
            dist = np.random.uniform(200.0, 1500.0)
            ts = base_time + pd.Timedelta(days=days_offset, hours=hour, minutes=np.random.randint(0, 60))
            
            src_nodes.append(u_idx)
            dst_nodes.append(merchant_idx)
            edge_feats.append([
                np.log10(amount),
                hour / 23.0,
                day_of_week / 6.0,
                min(dist / 500.0, 1.0),
                online
            ])
            edge_labels.append(1) # Bipartite spike transaction is fraudulent
            timestamps.append(ts)

    # ----------------------------------------------------
    # 4. CONSTRUCT GRAPH DATA & DATAFRAMES
    # ----------------------------------------------------
    edge_index = np.vstack([src_nodes, dst_nodes]).astype(np.int64)
    edge_features = np.array(edge_feats, dtype=np.float32)
    edge_labels = np.array(edge_labels, dtype=np.int64)
    timestamps = np.array(timestamps)
    
    # NetworkX Graph Construction
    nx_graph = nx.DiGraph()
    for i in range(num_nodes):
        nx_graph.add_node(i, 
                          type='user' if node_types[i] == 0 else 'merchant',
                          balance=10**node_features[i, 0],
                          age=node_features[i, 1] * (1825 if node_types[i] == 0 else 3650),
                          location_risk=node_features[i, 2],
                          device_trust=node_features[i, 3],
                          behavioral_risk=node_features[i, 4])
                          
    for j in range(len(edge_labels)):
        u = edge_index[0, j]
        v = edge_index[1, j]
        nx_graph.add_edge(u, v,
                          amount=10**edge_features[j, 0],
                          hour=int(edge_features[j, 1] * 23),
                          day_of_week=int(edge_features[j, 2] * 6),
                          distance=edge_features[j, 3] * 500.0,
                          online=bool(edge_features[j, 4]),
                          is_fraud=bool(edge_labels[j]),
                          timestamp=timestamps[j])

    # Convert to Pandas DataFrames for Analytics and Dashboards
    df_nodes = pd.DataFrame({
        'node_id': np.arange(num_nodes),
        'node_type': ['User' if t == 0 else 'Merchant' for t in node_types],
        'balance': 10**node_features[:, 0],
        'account_age_days': node_features[:, 1] * np.where(node_types == 0, 1825.0, 3650.0),
        'location_risk': node_features[:, 2],
        'device_trust': node_features[:, 3],
        'risk_multiplier': node_features[:, 4]
    })
    
    df_transactions = pd.DataFrame({
        'transaction_id': np.arange(len(edge_labels)),
        'source_id': edge_index[0],
        'target_id': edge_index[1],
        'amount': 10**edge_features[:, 0],
        'hour': (edge_features[:, 1] * 23.0).astype(int),
        'day_of_week': (edge_features[:, 2] * 6.0).astype(int),
        'distance_km': edge_features[:, 3] * 500.0,
        'online_flag': edge_features[:, 4].astype(int),
        'is_fraud': edge_labels,
        'timestamp': timestamps
    })
    
    # Sort transactions chronologically
    df_transactions = df_transactions.sort_values(by='timestamp').reset_index(drop=True)
    
    # Re-align tensors and arrays to sorted chronological order (useful for live streams)
    sorted_indices = df_transactions['transaction_id'].values
    edge_index = edge_index[:, sorted_indices]
    edge_features = edge_features[sorted_indices]
    edge_labels = edge_labels[sorted_indices]
    
    # Update transaction_id to reflect chronological index
    df_transactions['transaction_id'] = np.arange(len(df_transactions))

    return {
        'node_features': node_features,
        'node_types': node_types,
        'edge_index': edge_index,
        'edge_features': edge_features,
        'edge_labels': edge_labels,
        'df_transactions': df_transactions,
        'df_nodes': df_nodes,
        'nx_graph': nx_graph
    }

if __name__ == "__main__":
    data = generate_transaction_graph()
    print("Graph generated successfully!")
    print("Nodes:", len(data['df_nodes']))
    print("Transactions (Edges):", len(data['df_transactions']))
    print("Fraud Rate: {:.2f}%".format(data['df_transactions']['is_fraud'].mean() * 100))
