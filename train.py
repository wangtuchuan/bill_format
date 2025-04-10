import pandas as pd
from utils import get_embedding_model, get_collection
import environ


env = environ.Env()
environ.Env.read_env()


def process_kb_entry(row, embedding_model):
    """处理知识库中的单个条目"""
    merchant = str(row["商家"]).strip()
    description = str(row.get("商品信息", "")).strip()
    category = str(row["分类"]).strip()

    entry_id = f"kb_{merchant}_{description}"
    text_to_embed = f"商户: {merchant}"
    if description:
        text_to_embed += f" 描述关键词: {description}"

    embedding = embedding_model.encode(text_to_embed).tolist()

    return {
        "embedding": embedding,
        "document": text_to_embed,
        "metadata": {
            "merchant_name": merchant,
            "description_pattern": description,
            "category": category,
        },
        "id": entry_id,
    }


def build_knowledge_base(kb_file, collection, embedding_model):
    """构建知识库"""
    print(f"Building knowledge base from {kb_file}...")
    try:
        kb_df = pd.read_csv(kb_file)
    except FileNotFoundError:
        print(f"Error: Knowledge base file '{kb_file}' not found.")
        return

    existing_ids = set(collection.get()["ids"])
    print(f"Found {len(existing_ids)} existing entries in the collection.")

    new_entries = []
    for index, row in kb_df.iterrows():
        entry = process_kb_entry(row, embedding_model)
        if entry["id"] not in existing_ids:
            new_entries.append(entry)
            existing_ids.add(entry["id"])

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(kb_df)} entries...")

    if new_entries:
        print(f"Adding {len(new_entries)} new entries to the collection...")
        collection.add(
            embeddings=[e["embedding"] for e in new_entries],
            documents=[e["document"] for e in new_entries],
            metadatas=[e["metadata"] for e in new_entries],
            ids=[e["id"] for e in new_entries],
        )
        print("New entries added successfully.")
    else:
        print("No new entries to add.")


def main():
    """主函数入口"""
    # 初始化各个组件
    embedding_model = get_embedding_model()
    collection = get_collection()

    # 构建知识库
    build_knowledge_base(
        kb_file=env("BILL_FILES_PATH") + "test.csv",
        collection=collection,
        embedding_model=embedding_model,
    )


if __name__ == "__main__":
    main()
