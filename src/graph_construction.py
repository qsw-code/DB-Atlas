import networkx as nx
from pyvis.network import Network



class GraphBuilder:
    @staticmethod
    def build_dataset_entity_graph(triples: List[Dict], top_entities: Optional[List[str]] = None, 
                                   candidate_datasets: Optional[List[Tuple]] = None,
                                   retrieval_path: Optional[Dict] = None) -> nx.DiGraph:

        edges = retrieval_path["edges"]
        hop_info = retrieval_path["hop_info"]

        ent_info = {}

        for src, dst, _ in edges:
            hop = hop_info.get(src, None)
            if hop is None:
                continue


            if dst not in ent_info or hop > ent_info[dst]:
                ent_info[dst] = hop

        G = nx.DiGraph()
        
        if not candidate_datasets or not retrieval_path:
            return G
        

        path_datasets = set()
        path_entities = set()
        path_edges = []  
        
        for dataset, entity, hop in retrieval_path['edges']:
            path_datasets.add(dataset)
            path_entities.add(entity)
            path_edges.append((dataset, entity, hop))
        

        if top_entities:
            path_entities.update(top_entities)
        

        GraphBuilder._add_nodes(G, path_entities, path_datasets, top_entities, retrieval_path.get('hop_info', {}), ent_info)
        

        GraphBuilder._add_path_edges(G, path_edges, triples)
        
        return G

    @staticmethod
    def _add_nodes(G: nx.DiGraph, entities: Set[str], datasets: Set[str], 
                   top_entities: Optional[List[str]] = None,
                   hop_info: Optional[Dict[str, int]] = None, ent_info: Optional[Dict[str, int]] = None):

        query_entity_set = set(top_entities) if top_entities else set()
        hop_info = hop_info or {}
        ent_info = ent_info or {}

    
    
        hop_colors = {
            0: "#E64B35",  
            1: "#4DBBD5",  
            2: "#91D1C2", 
        }
        ent_colors = {
            1: "#00A087", 
            2: "#F39B7F",  
        }
        

        for entity in entities:
            if entity in query_entity_set:

                G.add_node(
                    entity, 
                    label=entity, 
                    color="#FFD700",  
                    shape="ellipse", 
                    title=f"Match Entities: {entity}",
                    size=30,
                    borderWidth=3
                )
            else:
                hop = ent_info.get(entity, 0)
                color = ent_colors.get(hop, "#90EE90")  

                G.add_node(
                    entity, 
                    label=entity, 
                    color=color, 
                    shape="ellipse", 
                    title=f"Bridge Entities: {entity}",
                    size=20
                )
        

        for ds in datasets:
            hop = hop_info.get(ds, 0)
            color = hop_colors.get(hop, "#B0C4DE")  
            G.add_node(
                ds, 
                label=ds, 
                color=color, 
                shape="box", 
                title=f"Dataset: {ds}\nHops: {hop+1}-hop",
                size=22,
                borderWidth=2
            )

    @staticmethod
    def _add_path_edges(G: nx.DiGraph, path_edges: List[Tuple[str, str, int]], 
                       triples: List[Dict]):

        triple_dict = {}
        for triple in triples:
            key = (triple["subject"], triple["object"])
            triple_dict[key] = triple["predicate"]
        

        hop_edge_styles = {
            0: {"color": "#8491B4", "width": 3, "dashes": False},  
            1: {"color": "#8491B4", "width": 2.5, "dashes": False},  
            2: {"color": "#8491B4", "width": 2, "dashes": [5, 5]},  
            3: {"color": "#8491B4", "width": 1.5, "dashes": [5, 5]}, 
        }
        
        for dataset, entity, hop in path_edges:
            predicate = triple_dict.get((dataset, entity), "related_to")
            style = hop_edge_styles.get(hop, {"color": "gray", "width": 1, "dashes": False})
            
            G.add_edge(
                dataset, 
                entity, 
                color=style["color"],
                width=style["width"],
                dashes=style.get("dashes", False),
                arrows="to", 
                title=f"{predicate} ({hop}-hop)",
                label= ""  
            )


class GraphVisualizer:
    @staticmethod
    def visualize_graph(graph, output_path="graph.html"):
        net = Network(notebook=False, directed=True, height="750px", width="100%")
        net.from_nx(graph)
        net.set_options("""
        {
        "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
            "gravitationalConstant": -200,
            "centralGravity": 0.002,
            "springLength": 260,
            "springConstant": 0.03
            },
            "minVelocity": 0.5,
            "maxVelocity": 60,
            "stabilization": { "iterations": 600 },
            "collision": {
            "enabled": true,
            "radius": 10
            }
        },
        "nodes": {
            "font": { "size": 32, "face": "arial" },
            "borderWidth": 2,
            "shape": "dot",
            "scaling": {
            "min": 20,
            "max": 60
            },
            "shadow": true
        },
        "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
            "smooth": { "enabled": true, "type": "continuous" },
            "font": { "size": 18, "align": "middle" }
        },
        "interaction": {
            "hover": true,
            "navigationButtons": true
        }
        }
        """)
        html = net.generate_html(output_path)
        net.save_graph(output_path)
        return html