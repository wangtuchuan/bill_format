import csv
import chardet


EAT_CATEGORY = ["饭", "面", "粉", "汤", "包道", "饺", "快餐", "饼", "丰宜", "鲍师傅", "餐厅", "便利店", "寿司", "奶茶", "酸奶", "奶", "麦当劳", "肯德基"]
SHOPPING_CATEGORY = ["京东", "淘宝", "拼多多", "PSO Brand", "服饰"]


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
    if "滴滴" in row_obj["交易对方"]:
        result = "交通"
    elif "餐" in row_obj["商品"]:
        result = "三餐"
    elif any([x in row_obj["交易对方"] for x in EAT_CATEGORY]):
        result = "三餐"
    elif row_obj["交易对方"] == "Apple" or "iCloud" in row_obj["交易对方"]:
        result = "应用软件"
    elif "动物" in row_obj["交易对方"] or "宠物" in row_obj["交易对方"]:
        result = "宠物"
    elif "医院" in row_obj["交易对方"]:
        result = "医疗"
    elif "充值" in row_obj["交易对方"] or any([x in row_obj["交易对方"] for x in ["联通", "电信"]]):
        result = "充值缴费"
    elif any([x in row_obj["交易对方"] for x in SHOPPING_CATEGORY]):
        result = "购物"
    return result


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
    if result == "爱车养车":
        result = "汽车"
    if result == "交通出行":
        result = "交通"
    if result == "服饰装扮":
        result = "服饰"
    if result == "数码电器":
        result = "数码"
    if result == "日用百货":
        result = "日用"
    return result


def process_alipay_func(row, csv_writer):
    """阿里处理函数"""
    csv_writer.writerow({
        '日期': row['交易时间'],
        '类型': row['收/支'],
        '金额': row['金额'],
        '一级分类': clean_alipay_category(row),
        '二级分类': '',
        '账户1': clean_alipay_account(row['收/付款方式']),
        '账户2': '',
        '备注': row['交易对方']
    })


def process_wechat_func(row, csv_writer):
    """微信处理函数"""
    csv_writer.writerow({
        '日期': row['交易时间'],
        '类型': row['收/支'],
        '金额': row['金额(元)'].split("¥")[1],
        '一级分类': clean_wechat_category(row),
        '二级分类': '',
        '账户1': clean_wechat_account(row['支付方式']),
        '账户2': '',
        '备注': row['交易对方']
    })


def main(source_type, filename):
    process_func = {
        ALIPAY: process_alipay_func,
        WECHAT: process_wechat_func,
    }
    with open(filename, 'rb') as f:
        result = chardet.detect(f.read())
    with open(filename, 'r', encoding=result['encoding'], errors="ignore") as csv_file:
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
        dict_list = [dict(zip(header, row)) for row in data[1:]]
        with open(os.path.join(DIR_PATH, f"{source_type}.csv"), 'w',
                  newline='', encoding='utf-8') as new_file:
            fieldnames = ['日期', '类型', '金额', '一级分类', '二级分类', '账户1', '账户2', "备注"]
            csv_writer = csv.DictWriter(new_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for row in dict_list:
                tmp_row = {}
                # 清楚多余空格字符
                for k, v in row.items():
                    tmp_row[k.strip()] = v

                process_func.get(source_type)(tmp_row, csv_writer)


if __name__ == "__main__":
    import os
    import glob

    import platform

    DIR_PATH = ""
    # 指定目录
    if platform.system() == "Windows":
        DIR_PATH = "C:\\Users\wang7\\iCloudDrive\\bill_files\\"
    elif platform.system() == "Darwin":
        DIR_PATH = "/Users/wangzhen/Library/Mobile Documents/com~apple~CloudDocs/bill_files/"

    ALIPAY = "alipay"
    WECHAT = "wechat"

    # 查找指定目录下所有csv文件
    csv_files = glob.glob(DIR_PATH + "*.csv")

    # 按时间排序
    csv_files.sort(key=os.path.getmtime, reverse=True)

    newest_wechat_csv_file = ""
    newest_alipay_csv_file = ""
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        if filename.startswith("alipay_record") and not newest_alipay_csv_file:
            newest_alipay_csv_file = filename
            main(ALIPAY, csv_file)
        if filename.startswith("微信支付账单") and not newest_wechat_csv_file:
            newest_wechat_csv_file = filename
            main(WECHAT, csv_file)
