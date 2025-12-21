import requests
import openai

class LLMRecommender_openai:

    
    def __init__(self, api_url: str, model_id: str, api_key: str):
        self.api_url = api_url.rstrip("/") + "/chat/completions"  
        self.model_id = model_id
        self.api_key = api_key
    
    def generate_recommendation(self, top_k: int, query: str, top_entities: list, candidate_datasets: list, dataset2triples: dict) -> str:

        dataset_triples_text = ""
        max_datasets = min(len(candidate_datasets), top_k)
        for ds, _ in candidate_datasets[:max_datasets]:
            triples_list = dataset2triples.get(ds, [])
            if triples_list:
                triples_text = "\n".join([f"{t['subject']} -- {t['predicate']} --> {t['object']}" for t in triples_list])
                dataset_triples_text += f"\ndatasets: {ds}\n{triples_text}\n"

        prompt = f"""
                Scientific research query: "{query}"

                Matched entities:
                {top_entities}

                Candidate datasets with their triples:
                {dataset_triples_text}

                Task:
                Based on the Scientific research query, the matched entities, and the triples of each candidate dataset:

                1. Recommend datasets that are most relevant for scientific research related to the query.
                2. For each dataset, briefly explain its scientific relevance, e.g., type of data, research application, or connection to the matched entities.
                3. Output each dataset on a separate line in the following format:
                   1. Dataset Name: [Brief reason for scientific relevance] \n
                   2. Dataset Name: [Brief reason for scientific relevance] \n
                4. Include only datasets that are directly relevant; if there are more than {top_k}, output only the top {top_k}.
                5. Use clear, formal English suitable for academic contexts.                """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 10000
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=300
            )
            if response.status_code == 200:
                result = response.json()
                return self._remove_think_tags(
                    result['choices'][0]['message']['content']
                )
            else:
                st.error(
                    f"LLM request failed: {response.status_code} {response.text}"
                )
                return ""
        except requests.exceptions.RequestException as e:
            st.error(f"Network request error: {e}")
            return ""
        except Exception as e:
            st.error(f"Unknown error occurred: {e}")
            return ""

    @staticmethod
    def _remove_think_tags(text: str) -> str:
        
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return re.sub(r"\s+", " ", cleaned_text).strip()
