# Database-Retrieval-Recommendation

# 🌐 Access & Deployment

You can follow the steps below to run the system locally from scratch, or directly visit our pre-deployed search and visualization system at:
🔗 **[http://123.123.123.123:456](http://123.123.123.123:456)**



# 🚀 Running Steps

Follow the steps below to run the system from scratch：


## 🛠️ Step 1: Download the LLM-Embedding Model

First, download the **qwen3-8b-emb** and place it under the `llm_model/` directory.

### 📂 Example directory structure:
```text
llm_model/
└── qwen3-8b-emb/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```
---

## ⚙️ Step 2: Run the Embedding Pipeline

Next, run the embedding module to generate entity embeddings from the knowledge triples.

### 📋 This step will:
- Load triples from `data/triples.jsonl`
- Extract entity objects
- Compute embeddings using the Qwen model
- Save embeddings to disk for reuse

### 💻 Run:
```bash
python emb/QwenTripletEmbeddingSearch.py
```

### 📄 After successful execution, the following file will be generated:
```text
llm_model/
└── qwen_embeddings.pkl
```
*This file contains the cached entity embeddings and will be automatically loaded in later runs.*

---

## 🏃 Step 3: Run the Main Program

Once embeddings are ready, run the main pipeline.

### 🛠️ Run:
```bash
python main.py
```

### 🛠️ Or, if using the argparse interface:
```bash
python main.py \
  --query "climate change" \
  --top_k 100 \
  --rec_topk 5 \
  --api_url https://api.openai.com/v1 \
  --model_id gpt-4o \
  --api_key YOUR_API_KEY
```

### 🔍 System Workflow:
- Perform semantic entity matching using embeddings
- Retrieve candidate datasets via multi-hop diffusion
- Build and visualize the dataset–entity graph
- Generate dataset recommendations using the LLM

---

## 📊 Step 4: View Outputs

Check the generated results and interactive visualizations.

### 🖥️ Console output:
- Dataset rankings and statistics.

### 🌐 Graph visualization:
```text
graph.html
```
*Open **graph.html** in your browser to explore dataset–entity relationships.*

---

## 🔁 Re-running the System

If `qwen_embeddings.pkl` already exists, embeddings will be loaded automatically.

### ⚠️ No need to recompute embeddings unless:
1. The triples data changes.
2. A different embedding model is used.


