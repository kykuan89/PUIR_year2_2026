import re
import pandas as pd

# 你提供的 mapping（原樣放進來即可）
PREFIX_TO_COLLEGE = {
    # 外語學院
    "英": "外語學院",
    "日": "外語學院",
    "西": "外語學院",

    # 人文暨社會科學學院
    "中": "人文暨社會科學學院",
    "大傳": "人文暨社會科學學院",
    "台文": "人文暨社會科學學院",
    "法律": "人文暨社會科學學院",
    "社工": "人文暨社會科學學院",
    "社工原專": "人文暨社會科學學院",
    "犯防": "人文暨社會科學學院",
    "犯防原專": "人文暨社會科學學院",
    "生態": "人文暨社會科學學院",

    # 理學院
    "食營": "理學院",
    "應化": "理學院",
    "化科": "理學院",
    "財工": "理學院",
    "永續智慧": "理學院",

    # 管理學院
    "國企": "管理學院",
    "行銷與數位經營": "管理學院",
    "會計": "管理學院",
    "財金": "管理學院",
    "觀光": "管理學院",
    "經管進": "管理學院",

    # 資訊學院
    "資管": "資訊學院",
    "資工": "資訊學院",
    "資科": "資訊學院",
    "人工智慧": "資訊學院",
    "智慧媒體學程": "資訊學院",
    "晶片設計": "資訊學院",
    "國際資訊學士學位學程": "資訊學院",

    # 國際學院
    "寰宇管理學程": "國際學院",
    "寰宇外語教育": "國際學院",
}


def _normalize_class_name(class_name: str) -> str:
    """
    把 '英一A' / '大傳二B' 這種班級名整理成較乾淨的字首以便對照：
    - 去空白
    - 移除年級與班別尾碼（例如 一A / 二B / 1A / 2B）
    注意：不會破壞像 '行銷與數位經營' 這種完整名稱
    """
    s = str(class_name).strip()
    if not s or s.lower() == "nan":
        return ""

    # 常見格式：前面是系所縮寫，後面接 一A/二B/三A... 或 1A/2B...
    # 只移除「末尾」的年級+班別
    s = re.sub(r"(?:[一二三四五六七八九十]|[1-9])\s*[A-Z]?$", "", s)
    s = s.strip()
    return s


def add_college_column(
    df: pd.DataFrame,
    class_col: str = "班級",
    out_col: str = "學院",
    prefix_to_college: dict = None,
    unknown: str = "未分類",
) -> pd.DataFrame:
    """
    由 df[class_col] 推導 df[out_col]（學院）。
    - 最長 prefix 優先匹配（解決 社工 vs 社工原專 這種重疊）
    """
    if prefix_to_college is None:
        prefix_to_college = PREFIX_TO_COLLEGE

    if class_col not in df.columns:
        # 沒班級欄就不動
        return df

    # 最長 prefix 優先
    prefixes = sorted(prefix_to_college.keys(), key=len, reverse=True)

    def infer_one(x):
        base = _normalize_class_name(x)
        if not base:
            return unknown
        for p in prefixes:
            if base.startswith(p):
                return prefix_to_college[p]
        return unknown

    df[out_col] = df[class_col].apply(infer_one)
    return df
