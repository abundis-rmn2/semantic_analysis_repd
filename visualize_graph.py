import networkx as nx
from pyvis.network import Network
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_gml(input_file: str, output_html: str = "graph_visualization.html"):
    """
    Standalone visualizer for GML/GraphML files using PyVis.
    """
    if not os.path.exists(input_file):
        logger.error(f"File not found: {input_file}")
        return

    logger.info(f"Loading graph from {input_file}...")
    try:
        # Try GML first, then GraphML
        if input_file.endswith(".gml"):
            G = nx.read_gml(input_file)
        else:
            G = nx.read_graphml(input_file)
    except Exception as e:
        logger.error(f"Error reading graph: {e}")
        return

    logger.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # Initialize PyVis network
    net = Network(
        height="100vh", 
        width="100%", 
        bgcolor="#222222", 
        font_color="white", 
        select_menu=True, 
        filter_menu=True
    )

    # Set physics for better layout
    net.force_atlas_2based()

    # Define color palette for clusters
    colors = ["#FF4B4B", "#1C83E1", "#00C0F2", "#F0F2F6", "#FFD21F", "#EB5757", "#6FCF97", "#9B51E0"]

    # Add nodes
    for node, data in G.nodes(data=True):
        cluster_id = data.get('cluster', -1)
        # Use cluster_id to pick a color
        node_color = colors[int(cluster_id) % len(colors)] if cluster_id != -1 else "#808080"
        
        # Build tooltip (HTML supported)
        label = str(node)
        title = f"<b>ID:</b> {node}<br>"
        for key, value in data.items():
            title += f"<b>{key}:</b> {value}<br>"
            
        net.add_node(
            node, 
            label=label, 
            title=title, 
            color=node_color,
            size=15 + (10 * G.degree(node) / max(1, G.number_of_nodes())) # Scale size by degree
        )

    # Add edges
    for source, target, data in G.edges(data=True):
        weight = data.get('weight', 1.0)
        reasons = data.get('reasons', 'N/A')
        net.add_edge(
            source, 
            target, 
            value=weight, 
            title=f"Weight: {weight}<br>Reasons: {reasons}",
            color="rgba(200, 200, 200, 0.5)"
        )

    # Save visualization
    net.save_graph(output_html)
    logger.info(f"Visualization saved to: {os.path.abspath(output_html)}")
    print(f"\nDone! Open this file in your browser:\nfile://{os.path.abspath(output_html)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone GML/GraphML Visualizer")
    parser.add_argument("input", help="Path to the .gml or .graphml file")
    parser.add_argument("--output", default="graph_viz.html", help="Output HTML file name")
    
    args = parser.parse_args()
    visualize_gml(args.input, args.output)
