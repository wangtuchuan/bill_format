import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import os
import environ

CATEGORIES = [
    "餐饮",
    "购物",
    "日用",
    "数码",
    "应用软件",
    "住房",
    "交通",
    "娱乐",
    "医疗",
    "人情",
    "宠物",
    "旅行",
    "公益",
    "其他",
    "投资",
]

LLM_MODEL_NAME = "deepseek-v3-250324"

env = environ.Env()
environ.Env.read_env()


def init_embedding_model():
    """初始化 embedding 模型"""
    print("Loading embedding model...")
    # 使用一个更稳定的模型
    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    print("Embedding model loaded.")
    return embedding_model


def init_chromadb():
    """初始化 ChromaDB 数据库"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "bill_knowledge_base"

    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' loaded.")
    except:
        print(f"Creating collection '{collection_name}'...")
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Collection '{collection_name}' created.")

    return collection


def init_llm_client():
    """初始化 LLM 客户端"""
    llm_client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=env("ARK_API_KEY"),
    )
    return llm_client


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


def classify_bill_with_rag(
    merchant_name, description, amount, collection, embedding_model, k=3
):
    """使用 RAG 进行分类"""
    # 构建查询向量
    merchant_name = str(merchant_name).strip()
    description = str(description).strip()
    amount = str(amount).strip()  # 金额通常对分类作用不大，但可以包含

    # 1. 创建查询文本并生成嵌入
    query_text = f"商户: {merchant_name}"
    if description:
        query_text += f" 描述: {description}"
    query_embedding = embedding_model.encode(query_text).tolist()

    # 2. 在向量数据库中检索相似条目
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],  # 获取文档、元数据和距离
        )
    except Exception as e:
        print(f"Error during ChromaDB query: {e}")
        return None  # 或者返回一个默认/错误分类

    # 3. 构建 RAG Prompt
    retrieved_context = ""
    if results and results["ids"][0]:  # 检查是否有检索结果
        retrieved_context += "以下是知识库中可能相关的记录：\n"
        for i in range(len(results["ids"][0])):
            doc_meta = results["metadatas"][0][i]
            doc_text = results["documents"][0][i]
            distance = results["distances"][0][i]
            # 只使用高度相关的结果 (距离越小越相关，对于余弦相似度是 1-similarity)
            # 这里设置一个阈值，比如 0.5 (可调整)
            if distance < 0.5:
                retrieved_context += f"- {doc_text} (已知分类: {doc_meta.get('category', '未知')}, 相似度: {1-distance:.2f})\n"
            else:
                retrieved_context += f"- {doc_text} (已知分类: {doc_meta.get('category', '未知')}, 相关度较低: {1-distance:.2f})\n"  # 可以选择不显示低相关度的

    else:
        retrieved_context = "知识库中未找到高度相关的记录。\n"
    return retrieved_context


def classify_bill_with_llm(
    llm_client, retrieved_context, merchant_name, description, amount
):
    # --- Prompt 设计 ---
    prompt = f"""
    你是一个个人记账分类助手。请根据以下新的账单信息和知识库提供的相关记录，将这条账单分类到以下类别之一：
    {', '.join(CATEGORIES)}

    新的账单信息：
    商户名称：{merchant_name}
    交易描述：{description}
    金额：{amount}

    {retrieved_context}
    请仔细分析商户名称和描述，并参考知识库信息（如果相关度高），给出最合适的分类。请只输出类别名称，不要添加任何其他解释。

    分类结果：
    """

    # 打印 Prompt (用于调试)
    # print("--- Sending Prompt to LLM ---")
    # print(prompt)
    # print("-----------------------------")

    # 4. 调用 LLM API
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个精准的个人记账分类助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # 低温，让输出更稳定、确定
            max_tokens=50,  # 分类名称通常很短
        )
        predicted_category = response.choices[0].message.content.strip()

        # 5. 清理和验证 LLM 输出
        # 确保输出是我们定义的类别之一
        cleaned_category = None
        for cat in CATEGORIES:
            if cat in predicted_category:  # 做一个简单的包含性检查
                cleaned_category = cat
                break

        if cleaned_category:
            return cleaned_category
        else:
            print(
                f"Warning: LLM output '{predicted_category}' is not in defined categories. Raw output kept."
            )
            # 可以选择返回原始输出或一个“待定”分类
            return predicted_category  # 或者 return "待定分类"

    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None  # 或者返回一个默认/错误分类


def main(need_build_kb=False):
    """主函数入口"""
    # 初始化各个组件
    embedding_model = init_embedding_model()
    collection = init_chromadb()
    llm_client = init_llm_client()

    # 构建知识库
    if need_build_kb:
        build_knowledge_base(
            kb_file=env("BILL_FILES_PATH") + "test.csv",
            collection=collection,
            embedding_model=embedding_model,
        )

    # 测试分类
    merchant_name = "滴滴出行"
    description = "滴滴快车-张师傅-08月08日行程"
    amount = "10"
    retrieved_context = classify_bill_with_rag(
        merchant_name, description, amount, collection, embedding_model
    )
    classified_category = classify_bill_with_llm(
        llm_client, retrieved_context, merchant_name, description, amount
    )
    print(f"Classified category: {classified_category}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the bill classification model")
    parser.add_argument(
        "--build",
        type=bool,
        default=False,
        help="Whether to build the knowledge base",
    )
    args = parser.parse_args()
    main(args.build)
