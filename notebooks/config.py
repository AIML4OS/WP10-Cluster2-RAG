# Configurations for RAG testing

# Using older version of data with 5-digit NACE
train_path = "https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/Cluster2/Norway-RAG-data/train_norwaydata.parquet"
test_path = "https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/Cluster2/Norway-RAG-data/test_norwaydata.parquet"

embeddings_model = "NbAiLab/nb-sbert-base" #"paraphrase-multilingual-MiniLM-L12-v2"
chromadb_path = "./chroma_db"
collection_name_class = "standard-naering-class"
collection_name_obs = "standard-naering-obs"

llm_model_onyxia = {
    'url': "https://llm.lab.sspcloud.fr/api/v1/chat/completions",
    'name': "qwen3-6-35b-moe",
}