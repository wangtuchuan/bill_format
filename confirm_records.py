import argparse
import json
from typing import List, Dict, Optional, Set

from db_utils import BillDatabase
from utils import get_collection, process_kb_entry, get_embedding_model


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


def get_user_choice(prompt: str, valid_choices: List[str], default: str = "") -> str:
    """获取用户选择"""
    while True:
        choice = input(prompt).strip()
        if not choice and default:
            return default
        if choice in valid_choices:
            return choice
        print("无效的选项，请重试")


def select_category(categories: List[str]) -> Optional[str]:
    """选择分类"""
    print("\n可选分类：")
    for i, cat in enumerate(categories, 1):
        print(f"{i}. {cat}")

    while True:
        cat_choice = input("请选择新的分类编号 (直接回车取消): ").strip()
        if not cat_choice:  # 直接回车取消
            print("已取消修改")
            return None

        try:
            cat_choice = int(cat_choice)
            if 1 <= cat_choice <= len(categories):
                new_category = categories[cat_choice - 1]
                return new_category
            else:
                print("无效的编号，请重试")
        except ValueError:
            print("请输入有效的数字")


def process_entries_to_collection(collection, entries: List[Dict]) -> None:
    """处理并更新条目到知识库"""
    if not entries:
        print("没有需要添加到知识库的记录")
        return

    # 去重：使用集合更高效
    id_cache: Set[str] = set()
    unique_entries = [
        entry
        for entry in entries
        if entry["id"] not in id_cache and not id_cache.add(entry["id"])
    ]

    if unique_entries:
        collection.upsert(
            embeddings=[e["embedding"] for e in unique_entries],
            documents=[e["document"] for e in unique_entries],
            metadatas=[e["metadata"] for e in unique_entries],
            ids=[e["id"] for e in unique_entries],
        )
        print("记录已成功更新到知识库")
    else:
        print("没有需要添加到知识库的记录")


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

            choice = get_user_choice(
                "请输入选项 (1-4，直接回车确认当前分类): ",
                ["", "1", "2", "3", "4"],
                default="1",
            )

            if choice in ["", "1"]:
                # 确认当前分类
                db.confirm_record(record["id"], record["predicted_category"])
                print("已确认当前分类")
                break
            elif choice == "2":
                # 修改分类
                new_category = select_category(categories)
                if new_category:
                    db.confirm_record(record["id"], new_category)
                    print(f"已修改分类为: {new_category}")
                break
            elif choice == "3":
                # 跳过
                print("已跳过当前记录")
                break
            elif choice == "4":
                # 退出
                print("退出确认流程")
                return

    # 将确认的记录添加到 collection
    confirmed_records = db.get_confirmed_records()
    if confirmed_records:
        print(f"\n将 {len(confirmed_records)} 条确认的记录添加到知识库...")
        embedding_model = get_embedding_model()

        # 使用列表推导式处理记录
        entries = [
            process_kb_entry(
                embedding_model,
                str(record["merchant"]).strip(),
                str(record["description"]).strip(),
                str(record["confirmed_category"]).strip(),
            )
            for record in confirmed_records
        ]

        process_entries_to_collection(collection, entries)


def process_file_records(file_path: str, collection) -> None:
    """处理文件中的记录"""
    with open(file_path, "r", encoding="utf-8") as f:
        records = f.readlines()

    embedding_model = get_embedding_model()

    entries = [
        process_kb_entry(
            embedding_model, merchant.strip(), description.strip(), category.strip()
        )
        for line in records
        for category, merchant, description in [line.strip().split(",")]
    ]

    process_entries_to_collection(collection, entries)


def load_categories(categories_file: Optional[str] = None) -> List[str]:
    """加载分类列表"""
    if categories_file:
        with open(categories_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        from constants import CATEGORIES

        return CATEGORIES


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="处理待确认的账单记录")
    parser.add_argument("--categories", type=str, help="分类列表的JSON文件路径")
    parser.add_argument("--type", type=str, help="账单类型", choices=["file", "db"])
    parser.add_argument("--file", type=str, help="账单文件路径")
    args = parser.parse_args()

    # 加载分类
    categories = load_categories(args.categories)
    collection = get_collection()

    # 根据类型处理记录
    if args.type == "file":
        if not args.file:
            print("错误: 使用文件模式时必须提供文件路径")
            return
        process_file_records(args.file, collection)
    else:
        db = BillDatabase()
        confirm_records(db, collection, categories)


if __name__ == "__main__":
    main()
