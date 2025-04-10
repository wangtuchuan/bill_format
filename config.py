# 账单分类配置
class BillConfig:
    # 相似度阈值配置
    SIMILARITY_THRESHOLD = 0.9  # 知识库匹配的相似度阈值
    LOW_SIMILARITY_THRESHOLD = 0.5  # 低相似度阈值，用于显示提示信息

    # 账户清洗规则
    ALIPAY_ACCOUNT_RULES = {
        "4532": "建设银行(4532)",
        "7393": "招商银行(7393)",
        "5463": "广发银行(5463)",
        "default": "余额宝",
    }

    WECHAT_ACCOUNT_RULES = {"/": "微信零钱", "零钱": "微信零钱"}

    # 特殊交易规则
    ALIPAY_SPECIAL_RULES = [
        {
            "condition": {
                "收/支": "不计收支",
                "交易对方": "兴全基金管理有限公司",
                "商品说明": "收益发放",
            },
            "category": "投资",
        }
    ]

    # 账单字段映射
    ALIPAY_FIELD_MAPPING = {
        "date": "交易时间",
        "type": "收/支",
        "amount": "金额",
        "merchant": "交易对方",
        "description": "商品说明",
        "account": "收/付款方式",
        "status": "交易状态",
    }

    WECHAT_FIELD_MAPPING = {
        "date": "交易时间",
        "type": "收/支",
        "amount": "金额(元)",
        "merchant": "交易对方",
        "description": "商品",
        "account": "支付方式",
    }

    # 输出字段配置
    OUTPUT_FIELDS = [
        "日期",
        "类型",
        "金额",
        "一级分类",
        "二级分类",
        "账户1",
        "账户2",
        "备注",
    ]

    # 文件路径配置
    BILL_FILES_PATH = "bill_files"  # 账单文件存放路径
    OUTPUT_FILENAME = "total.csv"  # 输出文件名

    # RAG配置
    RAG_CONFIG = {
        "k": 3,  # 检索的相似记录数量
        "include": ["documents", "metadatas", "distances"],  # 检索包含的字段
    }

    # LLM配置
    LLM_CONFIG = {"temperature": 0.1, "max_tokens": 50}
