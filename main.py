import os
import json
import re
import argparse
from LLM_embbeding import TripletEmbeddingSearch
from database_retriever import DatasetRetriever
from LLM_recommender import LLMRecommender
from graph_construction import GraphBuilder,GraphVisualizer


#####################################################

CONFIG = {
    "TRIPLE_FILE": "data/database_triples.jsonl",
    "LINK_JSON_FILE": "data/database_link.json", 
    "DESC_JSONL_FILE": "data/descriptions.jsonl",
    "emb_file": "llm_model/qwen_embeddings.pkl",
    "emb_model_path": "llm_model/qwen3-8b-emb",
    "TOP_K_MATCH": 100,
    "MIN_DATASETS_FOR_ENTITY": 2,
    "GRAPH_HEIGHT": 750,
    "GRAPH_WIDTH": "100%",
}


@st.cache_data
def load_dataset_links(file_path: str) -> Dict[str, str]:

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return {d["dataset_name"]: d["Website"] for d in data}


@st.cache_data
def load_dataset_descriptions(file_path: str) -> Dict[str, str]:
    descriptions = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):

            item = json.loads(line.strip())
            descriptions[item["database_name"]] = item["description"]



@st.cache_data
def load_triples(file_path: str) -> Tuple[List[Dict], List[str], Dict[str, Set], Dict[str, List]]:
    triples = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):

            triple = json.loads(line.strip())
            if all(key in triple for key in ["subject", "predicate", "object"]):
                triples.append(triple)   


    entity2datasets = defaultdict(set)
    dataset2triples = defaultdict(list)
    for triple in triples:
        dataset2triples[triple["subject"]].append(triple)
        entity2datasets[triple["object"]].add(triple["subject"])
    entities = list(entity2datasets.keys())
    return triples, entities, dict(entity2datasets), dict(dataset2triples)



@st.cache_data
def initialize_data():
    dataset_links = load_dataset_links(CONFIG["LINK_JSON_FILE"])
    dataset_descriptions = load_dataset_descriptions(CONFIG["DESC_JSONL_FILE"])
    triples, entities, entity2datasets, dataset2triples = load_triples(CONFIG["TRIPLE_FILE"])
    return dataset_links, dataset_descriptions, triples, entities, entity2datasets, dataset2triples



@st.cache_resource
def load_search_system():
    search_system = TripletEmbeddingSearch(CONFIG["emb_model_path"])

    if os.path.exists(CONFIG["emb_file"]):
        print("Please wait...")
        search_system.load_embeddings(CONFIG["emb_file"])
    else:
        search_system.load_jsonl_file(CONFIG["TRIPLE_FILE"])
        search_system.extract_and_embed_objects()
        search_system.save_embeddings(CONFIG["emb_file"])

    return search_system



def parse_args():

    parser = argparse.ArgumentParser(description="Scientific Dataset Retrieval & Recommendation System" )

    parser.add_argument("--query",type=str,default="climate change",help="Research query text")

    parser.add_argument("--top_k",type=int,default=50,help="Number of top matched entities")

    parser.add_argument("--rec_topk",type=int,default=5,help="Number of top recommended datasets")

    parser.add_argument("--api_url",type=str,required=True,help="LLM API URL (OpenAI-compatible)")

    parser.add_argument("--model_id",type=str,required=True,help="LLM model ID")

    parser.add_argument("--api_key",type=str,required=True,help="LLM API key")

    parser.add_argument("--diffusion_order",type=int,default=2,help="Diffusion hop level")

    parser.add_argument("--diff_top_k",type=int,default=1,help="Neighbor limit per hop")

    return parser.parse_args()



def main():


    args = parse_args()

    query = args.query
    top_k = args.top_k
    rec_topk = args.rec_topk
    api_url = args.api_url
    model_id = args.model_id
    app_key = args.api_key
    diffusion_order = args.diffusion_order
    diff_top_k = args.diff_top_k

    search_system = load_search_system()
    dataset_links, dataset_descriptions, triples, entities, entity2datasets, dataset2triples = initialize_data()

    results = search_system.search(query, top_k=top_k, threshold=0.1)
    top_entities = [d['object'] for d in results]
    retriever = DatasetRetriever(entities, entity2datasets, dataset2triples)
    seed_datasets, candidate_triples, candidate_datasets, top_entities,retrieval_path  = retriever.retrieve_datasets(top_entities, 2, diffusion_order, diff_top_k)
    print(candidate_datasets)

    graph = GraphBuilder.build_dataset_entity_graph(candidate_triples, top_entities, candidate_datasets,retrieval_path)
    graph_html = GraphVisualizer.visualize_graph(graph)
    print(graph_html)

    recommender = LLMRecommender(api_url, model_id, app_key)
    recommendation = recommender.generate_recommendation(rec_topk, query, top_entities, seed_datasets, st.session_state['dataset2triples'])
    print(recommendation)


if __name__ == "__main__":
    main()
