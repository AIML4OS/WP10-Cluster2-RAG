# Functions for creating a database for using with RAG

import pandas as pd


def index_classes(df: pd.DataFrame, code_col: str, name_col: str, note_col: str, embedder, collection):
    """Index each class as its own chunk: 'CODE: description'.
    Items are chunked  per class rather than using a moving width
    """
    texts = (df[name_col] + " " + df[note_col]).tolist()
    ids   = df[code_col].astype(str).tolist()
    metas = df[[code_col, name_col, note_col]].to_dict(orient="records")

    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metas,
    )
    print(f"Indexed {len(texts)} classes.")

def index_observations(df: pd.DataFrame, embedder, collection, embed_batch_size: int = 64,
    chroma_batch_size: int = 2000):
    """Index each training observation by its combined activity/purpose text."""
    
    texts = (
        df['company_name'].fillna("") + " " + 
        df['company_activity'].fillna("") + " " + 
        df['company_purpose'].fillna("")
    ).str.strip().tolist()
    ids   = [str(i) for i in df.index]  # row index as unique ID
    metas = df[['nace_21_code', 'company_name', 'company_activity', 'company_purpose']].to_dict(orient="records")
    
    n = len(texts)
    
    for i in range(0, n, chroma_batch_size):
        end = i + chroma_batch_size

        batch_texts = texts[i:end]
        batch_ids = ids[i:end]
        batch_metas = metas[i:end]
        
        # Compute embeddings (batched internally)
        batch_embeddings = embedder.encode(
            batch_texts,
            batch_size=embed_batch_size,
            show_progress_bar=True
        ).tolist()

        
        # Insert into Chroma
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_metas,
        )
        print(f"Indexed {min(end, n)}/{n}")

def get_fs():
    """Return an S3FileSystem instance."""
    return s3fs.S3FileSystem(
        anon=False,
    )

def download_chroma_from_s3(bucket: str, s3_prefix: str, local_path: str, fs):
    """Download the chroma_db folder from S3 before use."""
    s3_path = f"{bucket}/{s3_prefix}"
    if fs.exists(s3_path):
        fs.get(s3_path, local_path, recursive=True)
        print(f"Downloaded chroma_db from s3://{s3_path}")
    else:
        print(f"No existing chroma_db found at s3://{s3_path}, starting fresh.")

def upload_chroma_to_s3(local_path: str, bucket: str, s3_prefix: str, fs):
    """Upload the chroma_db folder to S3 after writing."""
    s3_path = f"{bucket}/{s3_prefix}"
    fs.put(local_path, s3_path, recursive=True)
    print(f"Uploaded chroma_db to {s3_path}")
    