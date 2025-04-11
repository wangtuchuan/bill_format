import sqlite3
from typing import List, Dict
from datetime import datetime, timedelta


class BillDatabase:
    """账单数据库管理类"""

    def __init__(self, db_path: str = "bill_records.db"):
        """初始化数据库连接"""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 创建账单文件记录表
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    UNIQUE(filename)
                )
            """
            )
            # 创建账单记录表
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS bill_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    date TEXT NOT NULL,
                    type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    category TEXT NOT NULL,
                    sub_category TEXT,
                    account1 TEXT,
                    account2 TEXT,
                    note TEXT,
                    merchant TEXT,
                    description TEXT,
                    UNIQUE(source_type, date, merchant, amount)
                )
            """
            )
            # 创建待确认记录表
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    date TEXT NOT NULL,
                    type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    merchant TEXT NOT NULL,
                    description TEXT,
                    predicted_category TEXT NOT NULL,
                    similarity REAL,
                    created_at TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    confirmed_category TEXT,
                    confirmed_at TEXT
                )
            """
            )
            conn.commit()

    def is_file_processed(self, filename: str) -> bool:
        """检查文件是否已处理"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM processed_files WHERE filename = ?", (filename,)
            )
            return cursor.fetchone() is not None

    def get_processed_dates(self, source_type: str) -> List[str]:
        """获取已处理的日期范围"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT start_date, end_date FROM processed_files WHERE source_type = ?",
                (source_type,),
            )
            return cursor.fetchall()

    def save_processed_file(
        self, filename: str, source_type: str, start_date: str, end_date: str
    ):
        """保存已处理的文件信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO processed_files 
                (filename, source_type, start_date, end_date, processed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    source_type,
                    start_date,
                    end_date,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def save_bill_record(self, record: Dict):
        """保存账单记录"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO bill_records 
                (source_type, date, type, amount, category, sub_category, 
                 account1, account2, note, merchant, description, filename)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["source_type"],
                    record["date"],
                    record["type"],
                    record["amount"],
                    record["category"],
                    record["sub_category"],
                    record["account1"],
                    record["account2"],
                    record["note"],
                    record["merchant"],
                    record["description"],
                    record["filename"],
                ),
            )
            conn.commit()

    def get_existing_records(
        self, source_type: str, start_date: str, end_date: str
    ) -> List[Dict]:
        """获取指定日期范围内的已存在记录"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM bill_records 
                WHERE source_type = ? 
                AND date BETWEEN ? AND ?
                """,
                (source_type, start_date, end_date),
            )
            return [dict(row) for row in cursor.fetchall()]

    def clear_old_records(self, days: int = 30):
        """清理指定天数前的记录"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            cursor.execute(
                "DELETE FROM processed_files WHERE processed_at < ?", (cutoff_date,)
            )
            conn.commit()

    def save_pending_record(
        self,
        source_type: str,
        date: str,
        type: str,
        amount: str,
        merchant: str,
        description: str,
        predicted_category: str,
        similarity: float,
    ) -> int:
        """保存待确认的记录"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO pending_records 
                (source_type, date, type, amount, merchant, description, 
                 predicted_category, similarity, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_type,
                    date,
                    type,
                    amount,
                    merchant,
                    description,
                    predicted_category,
                    similarity,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_pending_records(self, status: str = "pending") -> List[Dict]:
        """获取待确认的记录"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM pending_records WHERE status = ? ORDER BY created_at DESC",
                (status,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def confirm_record(self, record_id: int, confirmed_category: str) -> bool:
        """确认记录的分类"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE pending_records 
                SET status = 'confirmed',
                    confirmed_category = ?,
                    confirmed_at = ?
                WHERE id = ?
                """,
                (confirmed_category, datetime.now().isoformat(), record_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_confirmed_records(self) -> List[Dict]:
        """获取已确认的记录"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM pending_records WHERE status = 'confirmed'")
            return [dict(row) for row in cursor.fetchall()]
