from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import environ
from constants import EMBEDDING_MODEL_NAME, DB_PATH, COLLECTION_NAME, ARK_API_BASE_URL

env = environ.Env()
environ.Env.read_env()


def get_embedding_model():
    """初始化 embedding 模型"""
    print("Loading embedding model...")
    # 使用一个更稳定的模型
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")
    return embedding_model


def get_collection():
    """初始化 ChromaDB 数据库"""
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' loaded.")
    except Exception:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Collection '{COLLECTION_NAME}' created.")

    return collection


def get_llm_client():
    """初始化 LLM 客户端"""
    llm_client = OpenAI(
        base_url=ARK_API_BASE_URL,
        api_key=env("ARK_API_KEY"),
    )
    return llm_client
