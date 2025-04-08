import csv
import chardet

EAT_CATEGORY = [
    "饭",
    "面",
    "粉",
    "汤",
    "包道",
    "饺",
    "快餐",
    "饼",
    "丰宜",
    "鲍师傅",
    "餐厅",
    "便利店",
    "寿司",
    "奶茶",
    "酸奶",
    "奶",
    "麦当劳",
    "肯德基",
    "吐司",
    "大弗兰",
    "广东罗森",
    "喜市多",
    "智能货柜",
    "餐饮",
    "食品",
    "小荔园",
    "都城",
    "大阪王将",
]
SHOPPING_CATEGORY = ["京东", "淘宝", "拼多多", "PSO Brand", "服饰"]

ALIPAY = "alipay"
WECHAT = "wechat"

ALL_CATEGORIES = [
    "餐饮",
    "购物",
    "日用",
    "数码",
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


def format_bill_category(category):
    """统一过滤最终的分类"""
    return category if category in ALL_CATEGORIES else "其他"


def clean_wechat_account(obj):
    """清洗微信账户"""
    result = obj
    if obj == "/":
        result = "微信零钱"
    if obj == "零钱":
        result = "微信零钱"
    return result


def clean_wechat_category(row_obj):
    """清洗微信的分类"""
    result = row_obj["交易类型"]
    if "滴滴" in row_obj["交易对方"] or "中铁网络" in row_obj["交易对方"]:
        result = "交通"
    elif "超市" in row_obj["交易对方"]:
        result = "日用"
    elif "餐" in row_obj["商品"]:
        result = "餐饮"
    elif "餐饮" in row_obj["交易对方"]:
        result = "餐饮"
    elif any([x in row_obj["交易对方"] for x in EAT_CATEGORY]):
        result = "餐饮"
    elif row_obj["交易对方"] == "Apple" or "iCloud" in row_obj["交易对方"]:
        result = "娱乐"
    elif "动物" in row_obj["交易对方"] or "宠物" in row_obj["交易对方"]:
        result = "宠物"
    elif "医院" in row_obj["交易对方"]:
        result = "医疗"
    elif "充值" in row_obj["交易对方"] or any(
        [x in row_obj["交易对方"] for x in ["联通", "电信"]]
    ):
        result = "日用"
    elif any([x in row_obj["交易对方"] for x in SHOPPING_CATEGORY]):
        result = "购物"
    elif "思源工程" in row_obj["交易对方"]:
        result = "公益"
    elif "微信红包" in result:
        result = "其他"
    elif result == "商户消费":
        result = "其他"
    if "美团外卖" in row_obj["商品"]:
        result = "餐饮"
    return format_bill_category(result)


def clean_alipay_account(obj):
    """清洗微信账户"""
    result = "余额宝"
    if "4532" in obj:
        result = "建设银行(4532)"
    elif "7393" in obj:
        result = "招商银行(7393)"
    elif "5463" in obj:
        result = "广发银行(5463)"
    return result


def clean_alipay_category(row_obj):
    """清洗微信的分类"""
    result = row_obj["交易分类"].strip()
    if result == "餐饮美食":
        result = "餐饮"
    if result == "交通出行":
        result = "交通"
    if result in ["服饰装扮", "家居家装"]:
        result = "购物"
    if result == "数码电器":
        result = "数码"
    if result in ["生活服务", "日用百货", "爱车养车", "美容美发"]:
        result = "日用"
    if result == "医疗健康":
        result = "医疗"
    if result == "文化休闲":
        result = "娱乐"
    if result == "充值缴费":
        result = "日用"
    if "美团外卖" in row_obj["商品说明"]:
        result = "餐饮"
    return format_bill_category(result)


def process_alipay_func(row, csv_writer):
    """阿里处理函数"""
    # 余额宝每日收益
    if (
        row["收/支"].strip() == "不计收支"
        and row["交易对方"].strip() == "兴全基金管理有限公司"
        and "收益发放" in row["商品说明"]
    ):
        row["收/支"] = "收入"
        row["交易分类"] = "投资"
    if row["收/支"].strip() == "不计收支" and row["交易状态"].strip() == "退款成功":
        row["收/支"] = "收入"
    csv_writer.writerow(
        {
            "分类": clean_alipay_category(row),
            "商家": row["交易对方"],
            "商品信息": (
                row["商品说明"].split("-")[0]
                if "美团" in row["商品说明"]
                else row["商品说明"]
            ),
        }
    )


def process_wechat_func(row, csv_writer):
    """微信处理函数"""
    csv_writer.writerow(
        {
            "分类": clean_wechat_category(row),
            "商家": row["交易对方"],
            "商品信息": (
                row["商品"].split("-")[0] if "美团" in row["商品"] else row["商品"]
            ),
        }
    )


def load_data(source_type, filename):
    with open(filename, "rb") as f:
        result = chardet.detect(f.read())
    with open(filename, "r", encoding=result["encoding"], errors="ignore") as csv_file:
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
        # 将每一行数据转换为字典，并添加到列表中
        return [dict(zip(header, row)) for row in data[1:]]


def write_data(path, data):
    process_func = {
        ALIPAY: process_alipay_func,
        WECHAT: process_wechat_func,
    }
    with open(
        os.path.join(path, "test.csv"), "w", newline="", encoding="utf-8"
    ) as new_file:
        fieldnames = [
            "分类",
            "商家",
            "商品信息",
        ]
        csv_writer = csv.DictWriter(new_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        for source_type, dict_list in data.items():
            for row in dict_list:
                tmp_row = {}
                # 清楚多余空格字符
                for k, v in row.items():
                    tmp_row[k.strip()] = v
                process_func.get(source_type)(tmp_row, csv_writer)


def main():
    DIR_PATH = ""
    # 指定目录
    if platform.system() == "Windows":
        DIR_PATH = "Y:\\bill_files\\"
        # DIR_PATH = "C:\\Users\wang7\\iCloudDrive\\bill_files\\"
    elif platform.system() == "Darwin":
        DIR_PATH = (
            "/Users/wangzhen/Library/Mobile Documents/com~apple~CloudDocs/bill_files/"
        )

    # 查找指定目录下所有csv文件
    csv_files = glob.glob(DIR_PATH + "*.csv")

    # 按时间排序
    csv_files.sort(key=os.path.getmtime, reverse=True)

    data = {
        ALIPAY: [],
        WECHAT: [],
    }
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        if filename.startswith("alipay_record"):
            data[ALIPAY].extend(load_data(ALIPAY, csv_file))
        if filename.startswith("微信支付账单"):
            data[WECHAT].extend(load_data(WECHAT, csv_file))
    write_data(DIR_PATH, data)


if __name__ == "__main__":
    import os
    import glob

    import platform

    main()
