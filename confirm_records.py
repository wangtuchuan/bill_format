from db_utils import BillDatabase
from utils import get_collection
import argparse
from typing import List, Dict
import json


def display_record(record: Dict) -> None:
    """显示记录信息"""
    print("\n" + "=" * 50)
    print(f"ID: {record['id']}")
    print(f"日期: {record['date']}")
    print(f"类型: {record['type']}")
    print(f"金额: {record['amount']}")
    print(f"商户: {record['merchant']}")
    print(f"描述: {record['description']}")
    print(f"预测分类: {record['predicted_category']}")
    print(f"相似度: {record['similarity']:.2f}")
    print("=" * 50 + "\n")


def confirm_records(db: BillDatabase, collection, categories: List[str]) -> None:
    """处理待确认的记录"""
    pending_records = db.get_pending_records()

    if not pending_records:
        print("没有待确认的记录")
        return

    print(f"发现 {len(pending_records)} 条待确认的记录")

    for record in pending_records:
        display_record(record)

        while True:
            print("请选择操作：")
            print("1. 确认当前分类 (直接回车)")
            print("2. 修改分类")
            print("3. 跳过")
            print("4. 退出")

            choice = input("请输入选项 (1-4，直接回车确认当前分类): ").strip()

            if choice == "" or choice == "1":
                # 确认当前分类
                db.confirm_record(record["id"], record["predicted_category"])
                print("已确认当前分类")
                break
            elif choice == "2":
                # 修改分类
                print("\n可选分类：")
                for i, cat in enumerate(categories, 1):
                    print(f"{i}. {cat}")

                while True:
                    try:
                        cat_choice = input(
                            "请选择新的分类编号 (直接回车取消): "
                        ).strip()
                        if not cat_choice:  # 直接回车取消
                            print("已取消修改")
                            break

                        cat_choice = int(cat_choice)
                        if 1 <= cat_choice <= len(categories):
                            new_category = categories[cat_choice - 1]
                            db.confirm_record(record["id"], new_category)
                            print(f"已修改分类为: {new_category}")
                            break
                        else:
                            print("无效的编号，请重试")
                    except ValueError:
                        print("请输入有效的数字")
                break
            elif choice == "3":
                # 跳过
                print("已跳过当前记录")
                break
            elif choice == "4":
                # 退出
                print("退出确认流程")
                return
            else:
                print("无效的选项，请重试")

    # 将确认的记录添加到 collection
    confirmed_records = db.get_confirmed_records()
    if confirmed_records:
        print(f"\n将 {len(confirmed_records)} 条确认的记录添加到知识库...")

        # 获取 embedding 模型
        from utils import get_embedding_model

        embedding_model = get_embedding_model()

        # 处理每条记录
        entries = []
        for record in confirmed_records:
            merchant = str(record["merchant"]).strip()
            description = str(record["description"]).strip()
            category = str(record["confirmed_category"]).strip()

            entry_id = f"kb_{merchant}_{description}"
            text_to_embed = f"商户: {merchant}"
            if description:
                text_to_embed += f" 描述关键词: {description}"

            embedding = embedding_model.encode(text_to_embed).tolist()

            entries.append(
                {
                    "embedding": embedding,
                    "document": text_to_embed,
                    "metadata": {
                        "merchant_name": merchant,
                        "description_pattern": description,
                        "category": category,
                    },
                    "id": entry_id,
                }
            )

        # 添加到 collection
        if entries:
            # 获取所有要添加的 ID
            entry_ids = [e["id"] for e in entries]

            # 检查哪些 ID 已存在
            existing_ids = set(collection.get()["ids"])
            ids_to_update = set(entry_ids) & existing_ids
            ids_to_add = set(entry_ids) - existing_ids

            # 分别处理需要更新和新增的记录
            entries_to_update = [e for e in entries if e["id"] in ids_to_update]
            entries_to_add = [e for e in entries if e["id"] in ids_to_add]

            # 使用 update 方法统一处理所有记录
            print(
                f"更新 {len(entries_to_update)} 条已存在的记录，添加 {len(entries_to_add)} 条新记录..."
            )
            # 根据id 字段去除entries中重复的数据
            t_entries = []
            id_cache = set()
            for e in entries:
                if e["id"] not in id_cache:
                    t_entries.append(e)
                    id_cache.add(e["id"])

            collection.upsert(
                embeddings=[e["embedding"] for e in t_entries],
                documents=[e["document"] for e in t_entries],
                metadatas=[e["metadata"] for e in t_entries],
                ids=[e["id"] for e in t_entries],
            )

            print("确认的记录已成功更新到知识库")
        else:
            print("没有需要添加到知识库的记录")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="处理待确认的账单记录")
    parser.add_argument("--categories", type=str, help="分类列表的JSON文件路径")
    args = parser.parse_args()

    # 加载分类
    if args.categories:
        with open(args.categories, "r", encoding="utf-8") as f:
            categories = json.load(f)
    else:
        from constants import CATEGORIES

        categories = CATEGORIES

    # 初始化组件
    db = BillDatabase()
    collection = get_collection()

    # 处理待确认记录并更新知识库
    confirm_records(db, collection, categories)


if __name__ == "__main__":
    main()
