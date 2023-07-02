import networkx as nx
from src.utils import *

def split_tkg_by_timestamp(tkg):
    """Split a Temporal Knowledge Graph into subgraphs for each timestamp.

    Args:
    tkg: A NetworkX graph representing a Temporal Knowledge Graph, where each edge has an associated 'timestamp' attribute.

    Returns:
    A dictionary where each key is a timestamp and the value is the corresponding subgraph.
    """
    # Get the unique timestamps in the TKG
    timestamps = set(nx.get_edge_attributes(tkg, 'timestamp').values())

    # Create a subgraph for each timestamp
    subgraphs = {timestamp: nx.Graph() for timestamp in timestamps}

    # Add nodes and edges to each subgraph
    for u, v, data in tkg.edges(data=True):
        timestamp = data['timestamp']
        subgraphs[timestamp].add_edge(u, v, **data)

    # Return the dictionary of subgraphs
    return subgraphs




tkg = load_data()

# Add some nodes and edges with timestamps
tkg.add_edge('A', 'B', timestamp=1)
tkg.add_edge('B', 'C', timestamp=1)
tkg.add_edge('C', 'D', timestamp=2)
tkg.add_edge('D', 'E', timestamp=2)
tkg.add_edge('A', 'D', timestamp=3)

# Split the TKG into subgraphs by timestamp
subgraphs = split_tkg_by_timestamp(tkg)

# Now you can loop over your subgraphs for incremental learning:
for timestamp, subgraph in subgraphs.items():
    print(f"Processing timestamp {timestamp}...")
    # Extract entities and relationships for this timestamp
    entities = list(subgraph.nodes())
    relationships = list(subgraph.edges(data=True))
    # Now you can process these entities and relationships
