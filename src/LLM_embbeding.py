import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import List, Tuple, Dict
import os
from tqdm import tqdm

class TripletEmbeddingSearch:
    
    def __init__(self, model_path=''):


        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        self.model.eval()


        
        self.triplets = []
        self.objects = []
        self.object_embeddings = None
        self.object_to_triplets = {}
        
    def get_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:

        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i + batch_size]
                

                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                

                outputs = self.model(**inputs)
                

                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                

                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                

                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
        
    def load_jsonl_file(self, file_path: str):


        self.triplets = []
        

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                triplet = json.loads(line)
                if all(key in triplet for key in ['subject', 'predicate', 'object']):
                    self.triplets.append(triplet)

    
    def extract_and_embed_objects(self):


        unique_objects = set()
        self.object_to_triplets = {}
        
        for i, triplet in enumerate(self.triplets):
            obj = triplet['object']
            unique_objects.add(obj)
            

            if obj not in self.object_to_triplets:
                self.object_to_triplets[obj] = []
            self.object_to_triplets[obj].append({
                'index': i,
                'subject': triplet['subject'],
                'predicate': triplet['predicate']
            })
        
        self.objects = sorted(list(unique_objects))  

        
        

        self.object_embeddings = self.get_embeddings(self.objects, batch_size=8)

        
    def save_embeddings(self, save_path: str):

        data = {
            'objects': self.objects,
            'embeddings': self.object_embeddings,
            'object_to_triplets': self.object_to_triplets,
            'triplets': self.triplets
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

        
    def load_embeddings(self, load_path: str):

        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.objects = data['objects']
        self.object_embeddings = data['embeddings']
        self.object_to_triplets = data['object_to_triplets']
        self.triplets = data['triplets']
        

        
    def search(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:

        

        query_embedding = self.get_embeddings([query], batch_size=1)
        

        similarities = cosine_similarity(query_embedding, self.object_embeddings)[0]
        

        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for i in sorted_indices[:top_k]:
            similarity = similarities[i]
            if similarity < threshold:
                break
                
            object_name = self.objects[i]
            related_triplets = self.object_to_triplets[object_name]
            
            results.append({
                'object': object_name,
                'similarity': float(similarity)
            })
        
        return results
    
    def print_search_results(self, results: List[Dict]):

        if not results:
            print("No matching results found")
            return
            
        print(f"\nFound {len(results)} matching results:")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Object: {result['object']}")
            print(f"   Similarity: {result['similarity']:.4f}")


    def get_stats(self):

        subjects = set(t['subject'] for t in self.triplets)
        predicates = set(t['predicate'] for t in self.triplets)
        
        print(f"  Total number of triplets: {len(self.triplets)}")
        print(f"  Number of unique subjects: {len(subjects)}")
        print(f"  Number of unique predicates: {len(predicates)}")
        print(f"  Number of unique objects: {len(self.objects) if self.objects else 0}")
        
        # Display the most common predicates
        predicate_counts = {}
        for t in self.triplets:
            pred = t['predicate']
            predicate_counts[pred] = predicate_counts.get(pred, 0) + 1
        
        sorted_preds = sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\nMost common predicates:")
        for pred, count in sorted_preds[:10]:
            print(f"  {pred}: {count}")
