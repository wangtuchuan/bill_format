from utils import get_collection, get_llm_client, get_embedding_model
from constants import CATEGORIES, LLM_MODEL_NAME
from config import BillConfig
from db_utils import BillDatabase
import glob
import os
import environ
import chardet
import csv
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# 加载环境变量
env = environ.Env()
environ.Env.read_env()

# 常量定义
ALIPAY = "alipay"
WECHAT = "wechat"


@dataclass
class BillRecord:
    """账单记录数据类"""

    date: str
    type: str
    amount: float
    category: str
    sub_category: str = ""
    account1: str = ""
    account2: str = ""
    note: str = ""


class BillProcessor:
    """账单处理类"""

    def __init__(self, collection, embedding_model, llm_client):
        """初始化账单处理器"""
        self.collection = collection
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.config = BillConfig()
        self.DIR_PATH = env("BILL_FILES_PATH", default=self.config.BILL_FILES_PATH)
        self.db = BillDatabase()
        # 用于记录当前处理的记录信息
        self.current_source_type = None
        self.current_date = None
        self.current_type = None

    def get_date_range(self, data: List[Dict], source_type: str) -> Tuple[str, str]:
        """获取账单数据的日期范围"""
        date_field = (
            self.config.ALIPAY_FIELD_MAPPING["date"]
            if source_type == ALIPAY
            else self.config.WECHAT_FIELD_MAPPING["date"]
        )
        dates = [row[date_field] for row in data]
        return min(dates), max(dates)

    def clean_alipay_account(self, account: str) -> str:
        """清洗支付宝账户名称"""
        for key, value in self.config.ALIPAY_ACCOUNT_RULES.items():
            if key in account:
                return value
        return self.config.ALIPAY_ACCOUNT_RULES["default"]

    def clean_wechat_account(self, account: str) -> str:
        """清洗微信账户名称"""
        return self.config.WECHAT_ACCOUNT_RULES.get(account, account)

    def clean_alipay_category(self, row: Dict) -> str:
        """清洗支付宝分类"""
        # 检查特殊规则
        for rule in self.config.ALIPAY_SPECIAL_RULES:
            if all(row.get(k) == v for k, v in rule["condition"].items()):
                return rule["category"]

        return self.classify_bill(
            row[self.config.ALIPAY_FIELD_MAPPING["merchant"]],
            row[self.config.ALIPAY_FIELD_MAPPING["description"]],
            row[self.config.ALIPAY_FIELD_MAPPING["amount"]],
        )

    def clean_wechat_category(self, row: Dict) -> str:
        """清洗微信分类"""
        return self.classify_bill(
            row[self.config.WECHAT_FIELD_MAPPING["merchant"]],
            row[self.config.WECHAT_FIELD_MAPPING["description"]],
            row[self.config.WECHAT_FIELD_MAPPING["amount"]],
        )

    def classify_bill(self, merchant: str, description: str, amount: str) -> str:
        """使用 RAG 和 LLM 对账单进行分类"""
        # 1. 使用 RAG 获取相关上下文
        retrieved_context, best_match = self.classify_bill_with_rag(
            merchant, description, k=self.config.RAG_CONFIG["k"]
        )

        # 如果找到高度相似的匹配，直接返回知识库中的分类
        if best_match and best_match["similarity"] > self.config.SIMILARITY_THRESHOLD:
            return best_match["category"]

        # 2. 否则使用 LLM 进行分类
        predicted_category = self.classify_bill_with_llm(
            retrieved_context, merchant, description, amount
        )

        # 记录需要确认的记录
        self.db.save_pending_record(
            source_type=self.current_source_type,
            date=self.current_date,
            type=self.current_type,
            amount=amount,
            merchant=merchant,
            description=description,
            predicted_category=predicted_category,
            similarity=best_match["similarity"] if best_match else 0.0,
        )

        return predicted_category

    def classify_bill_with_rag(
        self, merchant_name: str, description: str, k: int = 3
    ) -> tuple[str, Optional[Dict]]:
        """使用 RAG 获取相关上下文"""
        merchant_name = str(merchant_name).strip()
        description = str(description).strip()

        query_text = f"商户: {merchant_name}"
        if description:
            query_text += f" 描述: {description}"

        query_embedding = self.embedding_model.encode(query_text).tolist()

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=self.config.RAG_CONFIG["include"],
            )
        except Exception as e:
            print(f"Error during ChromaDB query: {e}")
            return "知识库查询失败。", None

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

                if distance < self.config.LOW_SIMILARITY_THRESHOLD:
                    retrieved_context += f"- {doc_text} (已知分类: {doc_meta.get('category', '未知')}, 相似度: {similarity:.2f})\n"
                else:
                    retrieved_context += f"- {doc_text} (已知分类: {doc_meta.get('category', '未知')}, 相关度较低: {similarity:.2f})\n"
        else:
            retrieved_context = "知识库中未找到高度相关的记录。\n"

        return retrieved_context, best_match

    def classify_bill_with_llm(
        self, retrieved_context: str, merchant_name: str, description: str, amount: str
    ) -> str:
        """使用 LLM 进行分类"""
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
        print(f"使用大模型进行分类, {merchant_name}: {description}")
        try:
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个精准的个人记账分类助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.LLM_CONFIG["temperature"],
                max_tokens=self.config.LLM_CONFIG["max_tokens"],
            )
            predicted_category = response.choices[0].message.content.strip()

            # 验证输出是否在预定义类别中
            for cat in CATEGORIES:
                if cat in predicted_category:
                    return cat

            print(
                f"Warning: LLM output '{predicted_category}' is not in defined categories."
            )
            return "其他"
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return "其他"

    def load_data(self, source_type: str, filename: str) -> List[Dict]:
        """加载账单数据"""
        # 检查文件是否已处理
        # if self.db.is_file_processed(filename):
        #     print(f"文件 {filename} 已处理过，跳过")
        #     return []

        with open(filename, "rb") as f:
            result = chardet.detect(f.read())

        with open(
            filename, "r", encoding=result["encoding"], errors="ignore"
        ) as csv_file:
            reader = csv.reader(csv_file)
            delimiter_count = 0
            for row in reader:
                if len(row) == 0:
                    continue
                if "---" in row[0]:
                    delimiter_count += 1
                    end_count = 2 if source_type == ALIPAY else 1
                    if delimiter_count == end_count:
                        break

            data = [row for row in reader]
            header = data[0]
            records = [dict(zip(header, row)) for row in data[1:]]

            # 获取日期范围
            start_date, end_date = self.get_date_range(records, source_type)

            # 检查是否有重叠记录
            existing_records = self.db.get_existing_records(
                source_type, start_date, end_date
            )
            if existing_records:
                print(f"发现 {len(existing_records)} 条已处理的记录，将跳过这些记录")
                # 过滤掉已存在的记录
                existing_keys = {
                    (r["date"], r["merchant"], r["amount"]) for r in existing_records
                }
                records = [
                    r
                    for r in records
                    if (
                        r[
                            (
                                self.config.ALIPAY_FIELD_MAPPING["date"]
                                if source_type == ALIPAY
                                else self.config.WECHAT_FIELD_MAPPING["date"]
                            )
                        ],
                        r[
                            (
                                self.config.ALIPAY_FIELD_MAPPING["merchant"]
                                if source_type == ALIPAY
                                else self.config.WECHAT_FIELD_MAPPING["merchant"]
                            )
                        ],
                        float(
                            r[
                                (
                                    self.config.ALIPAY_FIELD_MAPPING["amount"]
                                    if source_type == ALIPAY
                                    else self.config.WECHAT_FIELD_MAPPING["amount"]
                                )
                            ].split("¥")[1]
                            if source_type == WECHAT
                            else r[self.config.ALIPAY_FIELD_MAPPING["amount"]]
                        ),
                    )
                    not in existing_keys
                ]

            # 保存文件处理记录
            if records:
                self.db.save_processed_file(
                    os.path.basename(filename), source_type, start_date, end_date
                )

            return records

    def process_alipay_record(self, row: Dict, filename: str) -> BillRecord:
        """处理支付宝账单记录"""
        # 特殊处理
        if (
            row[self.config.ALIPAY_FIELD_MAPPING["type"]].strip() == "不计收支"
            and row[self.config.ALIPAY_FIELD_MAPPING["status"]].strip() == "退款成功"
        ):
            row[self.config.ALIPAY_FIELD_MAPPING["type"]] = "收入"

        # 保存当前记录信息，用于后续分类
        self.current_source_type = ALIPAY
        self.current_date = row[self.config.ALIPAY_FIELD_MAPPING["date"]]
        self.current_type = row[self.config.ALIPAY_FIELD_MAPPING["type"]]

        record = BillRecord(
            date=self.current_date,
            type=self.current_type,
            amount=float(row[self.config.ALIPAY_FIELD_MAPPING["amount"]]),
            category=self.clean_alipay_category(row),
            account1=self.clean_alipay_account(
                row[self.config.ALIPAY_FIELD_MAPPING["account"]]
            ),
            note=f"{row[self.config.ALIPAY_FIELD_MAPPING['merchant']]} {row[self.config.ALIPAY_FIELD_MAPPING['description']]}",
        )

        # 保存到数据库
        self.db.save_bill_record(
            {
                "source_type": ALIPAY,
                "date": record.date,
                "type": record.type,
                "amount": record.amount,
                "category": record.category,
                "sub_category": record.sub_category,
                "account1": record.account1,
                "account2": record.account2,
                "note": record.note,
                "merchant": row[self.config.ALIPAY_FIELD_MAPPING["merchant"]],
                "description": row[self.config.ALIPAY_FIELD_MAPPING["description"]],
                "filename": os.path.basename(filename),
            }
        )

        return record

    def process_wechat_record(self, row: Dict, filename: str) -> BillRecord:
        """处理微信账单记录"""
        # 保存当前记录信息，用于后续分类
        self.current_source_type = WECHAT
        self.current_date = row[self.config.WECHAT_FIELD_MAPPING["date"]]
        self.current_type = row[self.config.WECHAT_FIELD_MAPPING["type"]]

        record = BillRecord(
            date=self.current_date,
            type=self.current_type,
            amount=float(row[self.config.WECHAT_FIELD_MAPPING["amount"]].split("¥")[1]),
            category=self.clean_wechat_category(row),
            account1=self.clean_wechat_account(
                row[self.config.WECHAT_FIELD_MAPPING["account"]]
            ),
            note=f"{row[self.config.WECHAT_FIELD_MAPPING['merchant']]} {row[self.config.WECHAT_FIELD_MAPPING['description']]}",
        )

        # 保存到数据库
        self.db.save_bill_record(
            {
                "source_type": WECHAT,
                "date": record.date,
                "type": record.type,
                "amount": record.amount,
                "category": record.category,
                "sub_category": record.sub_category,
                "account1": record.account1,
                "account2": record.account2,
                "note": record.note,
                "merchant": row[self.config.WECHAT_FIELD_MAPPING["merchant"]],
                "description": row[self.config.WECHAT_FIELD_MAPPING["description"]],
                "filename": os.path.basename(filename),
            }
        )

        return record

    def write_data(
        self, path: str, data: Dict[str, List[Dict]], filenames: Dict[str, str]
    ) -> None:
        """写入处理后的数据
        Args:
            path: 输出文件路径
            data: 处理后的数据
            filenames: 源文件名映射，key为source_type，value为filename
        """
        # 从数据库获取所有记录
        all_records = []
        for source_type in [ALIPAY, WECHAT]:
            records = self.db.get_existing_records(
                source_type,
                datetime.now().replace(day=1).strftime("%Y-%m-%d"),  # 本月第一天
                datetime.now().strftime("%Y-%m-%d"),  # 今天
            )
            all_records.extend(records)

        # 按日期排序
        all_records.sort(key=lambda x: x["date"])

        # 写入文件
        with open(
            os.path.join(path, self.config.OUTPUT_FILENAME),
            "w",
            newline="",
            encoding="utf-8",
        ) as new_file:
            csv_writer = csv.DictWriter(new_file, fieldnames=self.config.OUTPUT_FIELDS)
            csv_writer.writeheader()

            for source_type, records in data.items():
                for row in records:
                    # 清理数据
                    row = {k.strip(): v for k, v in row.items()}

                    if source_type == ALIPAY:
                        bill_record = self.process_alipay_record(row, filenames[ALIPAY])
                    else:
                        bill_record = self.process_wechat_record(row, filenames[WECHAT])

                    csv_writer.writerow(
                        {
                            "日期": bill_record.date,
                            "类型": bill_record.type,
                            "金额": bill_record.amount,
                            "一级分类": bill_record.category,
                            "二级分类": bill_record.sub_category,
                            "账户1": bill_record.account1,
                            "账户2": bill_record.account2,
                            "备注": bill_record.note,
                        }
                    )


def main():
    """主函数"""
    # 初始化组件
    collection = get_collection()
    llm_client = get_llm_client()
    embedding_model = get_embedding_model()
    bill_processor = BillProcessor(collection, embedding_model, llm_client)

    # 获取账单文件路径
    DIR_PATH = env("BILL_FILES_PATH", default=bill_processor.config.BILL_FILES_PATH)

    # 查找最新的账单文件
    csv_files = glob.glob(os.path.join(DIR_PATH, "*.csv"))
    csv_files.sort(key=os.path.getmtime, reverse=True)

    # 处理账单数据
    data = {}
    filenames = {}
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        if filename.startswith("alipay_record") and ALIPAY not in data:
            data[ALIPAY] = bill_processor.load_data(ALIPAY, csv_file)
            filenames[ALIPAY] = csv_file
        elif filename.startswith("微信支付账单") and WECHAT not in data:
            data[WECHAT] = bill_processor.load_data(WECHAT, csv_file)
            filenames[WECHAT] = csv_file

    # 写入处理后的数据
    bill_processor.write_data(DIR_PATH, data, filenames)


if __name__ == "__main__":
    main()
