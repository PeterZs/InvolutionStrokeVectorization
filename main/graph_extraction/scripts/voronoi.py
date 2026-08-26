import shapeutils
import graph_clusters
import numpy as np
import networkx as nx
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define Polygon
    shape = np.array([[0,0], [10,0], [10,4], [6,4], [6,10], [0,10], [0,0]])

    # Generate Voronoi
    v_vor, e_vor = shapeutils.get_polygon_voronoi(shape, densification_distance=0.3)

    # Analyze Graph
    G = graph_clusters.annotated_voronoi_graph(v_vor, e_vor, shape)

    # Visualization
    plt.figure(figsize=(8, 8))
    pos = nx.get_node_attributes(G, 'pos')
    
    # Plot polygon boundary
    plt.plot(shape[:,0], shape[:,1], 'k-', lw=2, alpha=0.5)

    # Draw edges: Green for 'hairs' (accessible), Grey for the 'core skeleton'
    for u, v in G.edges():
        # Edge is a hair if both nodes are accessible
        if G.nodes[u]['accessible'] is not None and G.nodes[v]['accessible'] is not None:
            color, alpha, lw = 'lightgray', 0.8, 2
        else:
            color, alpha, lw = 'blue', 0.4, 1
        
        p1, p2 = pos[u], pos[v]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=alpha, linewidth=lw)

    # Draw nodes
    # Highlight boundary nodes as red squares
    b_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'boundary']
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes, node_shape='o', node_color='blue', node_size=15, label='Boundary Connection')
    nx.draw_networkx_nodes(G, pos, nodelist=b_nodes, node_shape='s', node_color='red', node_size=50, label='Boundary Connection')

    plt.axis('equal')
    plt.title("Polygon Medial Axis: Core (Blue) vs Accessible Hairs (Green)")
    plt.show()

    # Demonstrate attribute access
    sample_node = list(G.nodes)[-1]
    print(f"Node {sample_node} accessible from boundary node: {G.nodes[sample_node]['accessible']}")