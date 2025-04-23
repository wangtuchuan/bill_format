from utils import get_collection, get_embedding_model

collection = get_collection()
embedding_model = get_embedding_model()

merchant_name = "赵发达湖南砍肉粉"
description = "美团/大众点评点餐订单-04104719791853129538767"

query_text = f"商户: {merchant_name}"
if description:
    query_text += f" 描述: {description}"

query_embedding = embedding_model.encode(query_text).tolist()


results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)

retrieved_context = ""
best_match = None
if results and results["ids"][0]:
    retrieved_context += "以下是知识库中可能相关的记录：\n"
    for i in range(len(results["ids"][0])):
        doc_meta = results["metadatas"][0][i]
        doc_text = results["documents"][0][i]
        distance = results["distances"][0][i]
        similarity = 1 - distance

        # 记录最佳匹配
        if not best_match or similarity > best_match["similarity"]:
            best_match = {
                "category": doc_meta.get("category", "未知"),
                "similarity": similarity,
                "text": doc_text,
            }

        if distance < 0.5:
            retrieved_context += f"- {doc_text} (已知分类: {doc_meta.get('category', '未知')}, 相似度: {similarity:.2f})\n"
        else:
            retrieved_context += f"- {doc_text} (已知分类: {doc_meta.get('category', '未知')}, 相关度较低: {similarity:.2f})\n"
else:
    retrieved_context = "知识库中未找到高度相关的记录。\n"

print(retrieved_context, best_match)
