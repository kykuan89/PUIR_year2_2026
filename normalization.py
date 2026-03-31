import re
import pandas as pd

# 你提供的 mapping（原樣放進來即可）
# 依照「十八學群及其學類對照表」建立（可逐步補充同義詞/系名）
#學院	班級	前綴
#外語學院	英文系	英
#外語學院	日文系	日
#外語學院	西文系	西
#人社院	中文系	中
#人社院	社工系	社工
#人社院	台文系	台文
#人社院	法律系	法律
#人社院	大傳系	大傳
#人社院	生態系	生態
#人社院	法律原專	法律原專
#人社院	社工原專	社工原專
#理學院	財工系	財工
#理學院	應化系	應化
#理學院	食營系	食營
#理學院	化科系	化科
#理學院	永續環境與智慧科技學士學位學程	永續
#管理學院	行銷與數位經營學系	行銷
#管理學院	國企系	國企
#管理學院	會計系	會計
#管理學院	觀光系	觀光
#管理學院	財金系	財金
#資訊學院	資管系	資管
#資訊學院	資工系	資工
#資訊學院	人工智慧系	人工智慧
#資訊學院	資科系	資科
#國際學院	國際資訊學士學位學程	國際
#國際學院	寰宇外語教育學士學位學程	寰宇外語
#國際學院	寰宇管理學士學位學程	寰宇管理

PROGRAM_SPECS = [
    ("外語學院", "英文系", "英", ["英文"]),
    ("外語學院", "日文系", "日", ["日文"]),
    ("外語學院", "西文系", "西", ["西文"]),
    ("人文暨社會科學學院", "中文系", "中", ["中文"]),
    ("人文暨社會科學學院", "社工系", "社工", []),
    ("人文暨社會科學學院", "台文系", "台文", []),
    ("人文暨社會科學學院", "法律系", "法律", []),
    ("人文暨社會科學學院", "大傳系", "大傳", []),
    ("人文暨社會科學學院", "生態系", "生態", []),
    ("人文暨社會科學學院", "法律原住民專班", "法律原專", []),
    ("人文暨社會科學學院", "社工原住民專班", "社工原專", []),
    ("理學院", "財工系", "財工", []),
    ("理學院", "應化系", "應化", []),
    ("理學院", "食營系", "食營", []),
    ("理學院", "化科系", "化科", []),
    ("理學院", "永續環境與智慧科技學士學位學程", "永續", ["永續智慧"]),
    ("管理學院", "行銷與數位經營學系", "行銷", ["行銷與數位經營"]),
    ("管理學院", "國企系", "國企", []),
    ("管理學院", "會計系", "會計", []),
    ("管理學院", "觀光系", "觀光", []),
    ("管理學院", "財金系", "財金", []),
    ("資訊學院", "資管系", "資管", []),
    ("資訊學院", "資工系", "資工", []),
    ("資訊學院", "人工智慧系", "人工智慧", []),
    ("資訊學院", "資科系", "資科", []),
    ("國際學院", "國際資訊學士學位學程", "國際", ["國際資訊"]),
    ("國際學院", "寰宇外語教育學士學位學程", "寰宇外語", ["寰宇外語教育"]),
    ("國際學院", "寰宇管理學士學位學程", "寰宇管理", ["寰宇管理學程"]),
]

COLLEGE_ORDER = list(dict.fromkeys(college for college, _, _, _ in PROGRAM_SPECS))
DEPARTMENT_ORDER = [department for _, department, _, _ in PROGRAM_SPECS]
PREFIX_ORDER = [prefix for _, _, prefix, _ in PROGRAM_SPECS]
PREFIX_TO_COLLEGE = {prefix: college for college, _, prefix, _ in PROGRAM_SPECS}
PREFIX_TO_DEPARTMENT = {prefix: department for _, department, prefix, _ in PROGRAM_SPECS}

_MATCH_RULES = []
for college, department, prefix, aliases in PROGRAM_SPECS:
    for alias in [prefix, department, *aliases]:
        _MATCH_RULES.append((alias, college, department, prefix))

# 額外保留舊資料常見別名，避免既有資料分類失敗。
for alias, college, department, prefix in [
    ("犯防原專", "人文暨社會科學學院", "犯罪防治原住民專班", "犯防原專"),
    ("犯防", "人文暨社會科學學院", "犯罪防治學系", "犯防"),
    ("經管進", "管理學院", "經營管理進修學士班", "經管進"),
    ("智慧媒體學程", "資訊學院", "智慧媒體學程", "智慧媒體學程"),
    ("晶片設計", "資訊學院", "晶片設計學程", "晶片設計"),
]:
    PREFIX_TO_COLLEGE[alias] = college
    PREFIX_TO_DEPARTMENT[alias] = department
    _MATCH_RULES.append((alias, college, department, prefix))

_MATCH_RULES.sort(key=lambda row: len(row[0]), reverse=True)


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


def get_class_info(class_name: str, unknown: str = "未分類") -> dict:
    """Infer canonical prefix/department/college from a raw class label."""
    base = _normalize_class_name(class_name)
    if not base:
        return {"prefix": unknown, "department": unknown, "college": unknown}

    for alias, college, department, prefix in _MATCH_RULES:
        if base.startswith(alias):
            return {"prefix": prefix, "department": department, "college": college}

    return {"prefix": base, "department": unknown, "college": unknown}


def add_prefix_column(
    df: pd.DataFrame,
    class_col: str = "班級",
    out_col: str = "前綴",
    unknown: str = "未分類",
) -> pd.DataFrame:
    if class_col not in df.columns:
        return df

    df[out_col] = df[class_col].apply(lambda x: get_class_info(x, unknown=unknown)["prefix"])
    return df


def add_department_column(
    df: pd.DataFrame,
    class_col: str = "班級",
    out_col: str = "學系",
    unknown: str = "未分類",
) -> pd.DataFrame:
    if class_col not in df.columns:
        return df

    df[out_col] = df[class_col].apply(lambda x: get_class_info(x, unknown=unknown)["department"])
    return df


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

    def infer_one(x):
        info = get_class_info(x, unknown=unknown)
        return info["college"]

    df[out_col] = df[class_col].apply(infer_one)
    return df
