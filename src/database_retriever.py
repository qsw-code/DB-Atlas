
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set, Optional
import numpy as np

#####

class DatasetRetriever:


    def __init__(self, entities: List[str], entity2datasets: Dict[str, Set],
                 dataset2triples: Dict[str, List]):
        self.entities = entities
        self.entity2datasets = entity2datasets
        self.dataset2triples = dataset2triples
        

        self.dataset2entities = defaultdict(set)
        for entity, datasets in self.entity2datasets.items():
            for dataset in datasets:
                self.dataset2entities[dataset].add(entity)
                

        self.dataset_richness = {
            d: len(triples) for d, triples in self.dataset2triples.items()
        }

    def retrieve_datasets(self, top_entities: List[str], top_k: int = 5, 
                          hop_level: int = 3, neighbor_limit: int = 1) -> Tuple[List[Dict], List[Tuple], List[str], Dict]:

        sorted_datasets = self._calculate_dataset_scores(top_entities)
        

        seed_datasets = sorted_datasets
        

        if hop_level > 1:
            query_entity_set = set(top_entities)
            final_datasets, retrieval_path = self._expand_datasets_with_path(
                seed_datasets, 
                max_hops=hop_level, 
                query_entity_set=query_entity_set,
                neighbor_limit=neighbor_limit,
                initial_entities=top_entities
            )
        else:
            final_datasets = seed_datasets

            retrieval_path = {
                'edges': [],
                'hop_info': {}
            }
            for entity in top_entities:
                for dataset, _ in seed_datasets:
                    if dataset in self.entity2datasets.get(entity, set()):
                        retrieval_path['edges'].append((dataset, entity, 0))
                        retrieval_path['hop_info'][dataset] = 0


        candidate_triples = self._collect_candidate_triples(final_datasets)
        
        return sorted_datasets, candidate_triples, final_datasets, top_entities, retrieval_path
    
    def _calculate_dataset_scores(self, top_entities: List[str]) -> List[Tuple[str, float]]:

        weights = np.exp(-0.1 * np.arange(len(top_entities)))
        entity_weight_dict = dict(zip(top_entities, weights))
        
        dataset_scores = defaultdict(float)
        for entity, weight in entity_weight_dict.items():
            for dataset in self.entity2datasets.get(entity, set()):
                dataset_scores[dataset] += weight
        
        return sorted(dataset_scores.items(), key=lambda x: (-x[1], x[0]))

    def _expand_datasets_with_path(self, seed_datasets: List[Tuple[str, float]], 
                                   max_hops: int, 
                                   query_entity_set: Set[str],
                                   neighbor_limit: int,
                                   initial_entities: List[str]) -> Tuple[List[Tuple[str, float]], Dict]:


        visited_scores = {d: s for d, s in seed_datasets}
        queue = deque([(d, s, 0) for d, s in seed_datasets])  # 种子数据集为0阶
        

        retrieval_path = {
            'edges': [],
            'hop_info': {}
        }
        

        for entity in initial_entities:
            for dataset, _ in seed_datasets:
                if dataset in self.entity2datasets.get(entity, set()):
                    retrieval_path['edges'].append((dataset, entity, 0))
                    retrieval_path['hop_info'][dataset] = 0
        
        DECAY_FACTOR = 0.5

        while queue:
            current_dataset, current_score, level = queue.popleft()
            
            if level >= max_hops - 1:  
            
            next_level = level + 1
            next_score = current_score * DECAY_FACTOR


            contained_entities = self.dataset2entities.get(current_dataset, set())

            for bridge_entity in contained_entities:

                all_candidates = self.entity2datasets.get(bridge_entity, set())
                

                selected_candidates = self._select_best_neighbors(
                    candidates=all_candidates,
                    current_dataset=current_dataset,
                    query_entity_set=query_entity_set,
                    limit=neighbor_limit
                )
                
                for neighbor in selected_candidates:
                    if neighbor not in visited_scores or next_score > visited_scores[neighbor]:
                        is_new = neighbor not in visited_scores
                        visited_scores[neighbor] = next_score
                        

                        if is_new:
                            retrieval_path['edges'].append((neighbor, bridge_entity, next_level))
                            retrieval_path['hop_info'][neighbor] = next_level

                            if (current_dataset, bridge_entity, level) not in retrieval_path['edges']:
                                retrieval_path['edges'].append((current_dataset, bridge_entity, level))
                        
                        if next_level < max_hops - 1:
                            queue.append((neighbor, next_score, next_level))

        return sorted(visited_scores.items(), key=lambda x: (-x[1], x[0])), retrieval_path

    def _select_best_neighbors(self, candidates: Set[str], current_dataset: str, 
                               query_entity_set: Set[str], limit: int) -> List[str]:

        valid_candidates = [c for c in candidates if c != current_dataset]
        
        if len(valid_candidates) <= limit:
            return valid_candidates
        
        candidate_scores = []
        for cand in valid_candidates:
            cand_entities = self.dataset2entities.get(cand, set())
            hit_score = len(cand_entities.intersection(query_entity_set))
            richness_score = self.dataset_richness.get(cand, 0)
            candidate_scores.append((cand, hit_score, richness_score))
        
        sorted_candidates = sorted(candidate_scores, key=lambda x: (-x[1], -x[2]))
        return [x[0] for x in sorted_candidates[:limit]]
    
    def _collect_candidate_triples(self, sorted_datasets: List[Tuple[str, float]]) -> List[Dict]:
        candidate_triples = []
        for dataset, _ in sorted_datasets:
            candidate_triples.extend(self.dataset2triples.get(dataset, []))
        return candidate_triples
