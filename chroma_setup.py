import pandas as pd
import chromadb
from datasets import load_dataset

def setup_chroma_db(path="./chroma_db"):
    ds = load_dataset("deccan-ai/insuranceQA-v2")
    df = pd.concat([split.to_pandas() for split in ds.values()], ignore_index=True)
    df["combined"] = "Question: " + df["input"] + " \n Answer:  " + df["output"]

    df = df.sample(500, random_state=42).reset_index(drop=True)

    chroma_client = chromadb.PersistentClient(path=path)
    collection = chroma_client.get_or_create_collection(name="insurance_FAQ_collection")

    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        collection.add(
            documents=batch_df["combined"].tolist(),
            metadatas=[{"question": q, "answer": a} for q, a in zip(batch_df["input"], batch_df["output"])],
            ids=batch_df.index.astype(str).tolist()
        )

    print("✅ ChromaDB ready")
    return collection
