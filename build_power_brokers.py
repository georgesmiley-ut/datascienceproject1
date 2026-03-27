#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import igraph as ig


def build_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> ig.Graph:
    """Create a directed igraph graph from node and edge tables.

    Purpose:
        Build a graph with vertex names aligned to the `id` order in `nodes_df`
        so computed centrality values can be written back to the same rows.

    Args:
        nodes_df: DataFrame containing at least an `id` column.
        edges_df: DataFrame containing `source` and `target` columns.

    Returns:
        A directed igraph.Graph containing all nodes and edges from the inputs.
    """
    g = ig.Graph(directed=True)
    # Preserve node ordering from nodes_df so scores align back to rows
    g.add_vertices(nodes_df["id"].tolist())
    g.add_edges(list(zip(edges_df["source"], edges_df["target"])))
    return g


def directed_closeness(g: ig.Graph, mode: str = "out") -> list[float]:
    """Compute directed closeness centrality values for all vertices.

    Purpose:
        Calculate normalized closeness scores using igraph's directed mode
        options to quantify each site's network accessibility.

    Args:
        g: Directed graph whose vertex scores should be computed.
        mode: Closeness direction mode; one of `out`, `in`, or `all`.

    Returns:
        A list of closeness centrality values in graph vertex order.
    """
    return g.closeness(mode=mode, normalized=True)


def main() -> None:
    """Run the CLI pipeline for power broker score generation.

    Purpose:
        Load ORBIS node/edge data, compute directed closeness with all edges and
        with road edges removed, then write the enriched node table to CSV.

    Args:
        None.

    Returns:
        None. Writes an output CSV as a side effect.
    """
    parser = argparse.ArgumentParser(
        description="Build igraph graphs and compute directed closeness centrality."
    )
    parser.add_argument(
        "--nodes",
        default="/Users/caydenopperman/Documents/agent_coding/datascienceproject1/orbis_nodes_0514.csv",
        help="Path to node table CSV (default: orbis_nodes_0514.csv)",
    )
    parser.add_argument(
        "--edges",
        default="/Users/caydenopperman/Documents/agent_coding/datascienceproject1/orbis_edges_0514.csv",
        help="Path to edge table CSV (default: orbis_edges_0514.csv)",
    )
    parser.add_argument(
        "--out",
        default="orbis_nodes_0514_with_power_broker_scores.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--mode",
        default="out",
        choices=["out", "in", "all"],
        help="Directed closeness mode (default: out)",
    )
    args = parser.parse_args()

    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    # Ensure consistent string IDs for igraph name matching
    nodes_df["id"] = nodes_df["id"].astype(str)
    edges_df["source"] = edges_df["source"].astype(str)
    edges_df["target"] = edges_df["target"].astype(str)

    # Graph including all edges (including type=road)
    g_all = build_graph(nodes_df, edges_df)
    closeness_all = directed_closeness(g_all, mode=args.mode)

    # Graph excluding road edges
    edges_no_road = edges_df[edges_df["type"] != "road"]
    g_no_road = build_graph(nodes_df, edges_no_road)
    closeness_no_road = directed_closeness(g_no_road, mode=args.mode)

    out_df = nodes_df.copy()
    out_df["closeness_all_edges"] = closeness_all
    out_df["closeness_no_road_edges"] = closeness_no_road

    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
