import re
from dataclasses import dataclass
from typing import Literal, Optional, Dict, List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from normalization import add_college_column, PREFIX_TO_COLLEGE

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="問卷互動分析", layout="wide")

# 改成你的檔名 / 路徑
FILE_PATH = "114學年度大二學習投入問卷調查-去識別化.xlsx"
DEFAULT_SHEET = None  # None = 第一個sheet；或填 "Sheet1"

# 複選/排名題常見分隔符（你的資料多半是 ; / ；）
MULTI_SEP_REGEX = r"[;；]"

QuestionType = Literal[
    "meta",
    "single_choice",
    "multi_choice",
    "multi_choice_ranked",
    "likert",
    "short_answer",
    "unknown",
]

@dataclass(frozen=True)
class TagResult:
    qtype: QuestionType
    reason: str
    confidence: float

# -----------------------------
# Type tagger (header + values)
# -----------------------------
class SurveyColumnTypeTagger:
    DEFAULT_META_HINTS = [
        "ID", "開始時間", "完成時間", "上次修改時間", "填答時間", "IP", "學號", "姓名", "Email"
    ]

    LIKERT_SETS: List[set] = [
        {"是", "否", "不適用"},
        {"是 (Yes)", "否 (No)", "不適用 (Not applicable)"},
        {"滿意", "普通", "不滿意", "不適用"},
        {"滿意 (Agree)", "普通 (Neutral)", "不滿意 (Disagree)", "不適用 (Not applicable)"},
        {"滿意", "普通", "不滿意", "不清楚"},
        {"滿意 (Agree)", "普通 (Neutral)", "不滿意 (Disagree)", "不清楚 (Unclear)"},
        {"是", "否"},
    ]

    def __init__(self, extra_meta_cols: Optional[List[str]] = None):
        self.extra_meta_cols = set(extra_meta_cols or [])

    def tag_column(self, df: pd.DataFrame, col: str) -> TagResult:
        header = self._norm(col)

        # meta by name
        if col in self.extra_meta_cols:
            return TagResult("meta", "in user-provided meta list", 1.0)

        # meta by hint
        if any(h in header for h in self.DEFAULT_META_HINTS):
            return TagResult("meta", "header looks like metadata", 0.85)

        s = df[col]
        values = s.dropna().astype(str)
        sample = values.head(400)
        uniq = set(sample.unique())
        uniq_norm = {self._norm(v) for v in uniq}

        # ---- sparse column heuristic ----
        n_total = len(s)
        n_resp = len(values)
        resp_rate = n_resp / max(n_total, 1)

        u = len(uniq_norm)

        if resp_rate <= 0.5 and 2 <= u <= max(20, int(0.5 * n_resp)):
            return TagResult(
                "single_choice",
                f"sparse responses: resp_rate={resp_rate:.2f}, uniq={u}",
                0.85
            )

        # ranking hint (優先於一般複選)
        if self._has_any(header, ["需排序", "排序", "rank", "順位", "最主要", "第二次要", "第三次要"]):
            if self._looks_delimited_multi(sample):
                return TagResult("multi_choice_ranked", "header indicates ranking + delimited values", 0.95)
            return TagResult("multi_choice_ranked", "header indicates ranking", 0.8)

        # explicit multi-choice hint
        if self._has_any(header, ["可複選", "(可複選)", "multiple choice"]):
            if self._looks_delimited_multi(sample):
                return TagResult("multi_choice", "header says multi + delimited values", 0.97)
            return TagResult("multi_choice", "header says multi-choice", 0.9)

        # likert by value set
        for lk in self.LIKERT_SETS:
            if uniq_norm.issubset({self._norm(x) for x in lk}) and len(uniq_norm) >= 2:
                # 簡答題有時也會出現「不滿意」等字在題幹，但 value set 會很分散，不會 subset
                return TagResult("likert", "values match a likert option set", 0.98)

        # multi-choice by delimiter pattern
        if self._looks_delimited_multi(sample):
            return TagResult("multi_choice", "values look like multi selections delimited", 0.8)

        # open-ended hints
        if self._has_any(header, ["請提供", "意見", "建議", "最喜歡", "請列出", "課程名稱", "please provide", "please list", "feedback", "suggestions"]):
            return TagResult("short_answer", "header looks open-ended", 0.8)

        # categorical single-choice heuristic (improved)
        n = len(sample)
        u = len(uniq_norm)
        if n > 0:
            uniq_ratio = u / n
        else:
            uniq_ratio = 1.0

        avg_len = sample.astype(str).str.len().mean() if len(sample) else 0

        # 規則想法：
        # - 單選題通常 unique 不會太大（例如 <= 30 或 <= max(12, 0.2*n)）
        # - 且 uniq_ratio 不會太接近 1（不像簡答幾乎人人不同）
        # - 且平均字串長度通常不會很長（不像長句）
        max_u = min(30, max(12, int(0.2 * n)))  # 動態上限：樣本越多容忍越多類別
        if 2 <= u <= max_u and uniq_ratio <= 0.6:
            return TagResult(
                "single_choice",
                f"categorical pattern: uniq={u}, uniq_ratio={uniq_ratio:.2f}, avg_len={avg_len:.1f}",
                0.78
            )

        # free-text heuristic
        if self._looks_free_text(sample):
            return TagResult("short_answer", "values look like free text", 0.65)

        return TagResult("unknown", "no strong signal", 0.3)

    @staticmethod
    def _norm(x: str) -> str:
        x = str(x).strip()
        x = re.sub(r"\s+", " ", x)
        return x

    @staticmethod
    def _has_any(text: str, needles: List[str]) -> bool:
        return any(n in text for n in needles)

    @staticmethod
    def _looks_delimited_multi(values: pd.Series) -> bool:
        if len(values) == 0:
            return False
        v = values.astype(str)
        hit = v.str.contains(MULTI_SEP_REGEX, regex=True).mean()
        return hit >= 0.15

    @staticmethod
    def _looks_free_text(values: pd.Series) -> bool:
        if len(values) == 0:
            return False
        v = values.astype(str)
        avg_len = v.str.len().mean()
        uniq_ratio = v.nunique() / max(len(v), 1)
        return (avg_len >= 12 and uniq_ratio >= 0.5)

    @staticmethod
    def _looks_categorical(uniq: set) -> bool:
        joined = " ".join(list(uniq))
        patterns = ["小時", "以上", "0~", "1~", "11~", "21~", "over", "yes", "no", "滿意", "普通", "不滿意", "不適用", "不清楚"]
        joined_lower = joined.lower()
        return any(p.lower() in joined_lower for p in patterns)

# -----------------------------
# Data utils
# -----------------------------
@st.cache_data(show_spinner=False)
def load_excel(path: str, sheet: Optional[str]):
    if sheet:
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_excel(path)
    # drop all-empty columns
    df = df.dropna(axis=1, how="all")
    # strip object cols
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].apply(lambda c: c.astype(str).str.strip().replace({"nan": np.nan}))
    return df

def explode_multi(series: pd.Series) -> pd.Series:
    """Explode multi-select cells into one item per row (keeping index)."""
    s = series.dropna().astype(str)
    parts = s.str.split(MULTI_SEP_REGEX, regex=True)
    out = parts.explode().str.strip()
    out = out[out.notna() & (out != "")]
    return out

def ranked_stats(series: pd.Series, top_k: int = 3, weights: Optional[Dict[int, int]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
      - rank_count: option x rank table (count)
      - weighted_score: option weighted score (Borda)
    """
    if weights is None:
        weights = {1: 3, 2: 2, 3: 1}

    s = series.dropna().astype(str)
    rank_rows = []
    score = {}

    for cell in s:
        items = [x.strip() for x in re.split(MULTI_SEP_REGEX, cell) if x.strip()]
        items = items[:top_k]
        for r, opt in enumerate(items, start=1):
            rank_rows.append((opt, r))
            if r in weights:
                score[opt] = score.get(opt, 0) + weights[r]

    rank_df = pd.DataFrame(rank_rows, columns=["option", "rank"])
    if rank_df.empty:
        rank_table = pd.DataFrame()
    else:
        rank_table = (rank_df
                      .groupby(["option", "rank"])
                      .size()
                      .unstack(fill_value=0)
                      .reset_index())

    score_df = (pd.Series(score, name="weighted_score")
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={"index": "option"}))

    return rank_table, score_df

def ranked_score_table(series: pd.Series, top_k: int = 3, weights: Optional[Dict[int, int]] = None) -> pd.DataFrame:
    """
    回傳每個 option 的：
    - total_score：該組 Borda 總分
    - avg_score：該組 Borda 平均分（除以該組有效作答人數）
    """
    if weights is None:
        weights = {1: 3, 2: 2, 3: 1}

    s = series.dropna().astype(str)
    n_resp = len(s)
    if n_resp == 0:
        return pd.DataFrame(columns=["option", "total_score", "avg_score", "n_resp"])

    score = {}
    for cell in s:
        items = [x.strip() for x in re.split(MULTI_SEP_REGEX, cell) if x.strip()]
        items = items[:top_k]
        for r, opt in enumerate(items, start=1):
            if r in weights:
                score[opt] = score.get(opt, 0) + weights[r]

    out = (pd.Series(score, name="total_score")
             .sort_values(ascending=False)
             .reset_index()
             .rename(columns={"index": "option"}))
    out["n_resp"] = n_resp
    out["avg_score"] = out["total_score"] / n_resp
    return out

def group_value_counts(df: pd.DataFrame, col: str, group_col: Optional[str] = None) -> pd.DataFrame:
    if group_col and group_col in df.columns:
        tmp = df[[group_col, col]].dropna()
        out = tmp.groupby([group_col, col]).size().reset_index(name="count")
        return out
    else:
        out = df[col].dropna().value_counts().reset_index()
        out.columns = ["value", "count"]
        return out

def add_percent(df_counts: pd.DataFrame, value_col: str, count_col: str = "count", group_col: Optional[str] = None) -> pd.DataFrame:
    out = df_counts.copy()
    if group_col and group_col in out.columns:
        denom = out.groupby(group_col)[count_col].transform("sum")
        out["percent"] = (out[count_col] / denom) * 100
    else:
        out["percent"] = (out[count_col] / out[count_col].sum()) * 100
    return out


def parse_class_key(class_name: str, college_order=None):
    import re
    text = str(class_name or "").strip()
    if not text:
        return (len(college_order) if college_order is not None else 999, "", 0, "")

    # 從 PREFIX_TO_COLLEGE 匹配最長前綴決定學院
    college = "未分類"
    for p in sorted(PREFIX_TO_COLLEGE.keys(), key=len, reverse=True):
        if text.startswith(p):
            college = PREFIX_TO_COLLEGE[p]
            break

    college_rank = college_order.index(college) if college_order and college in college_order else (len(college_order) if college_order else 999)

    m = re.match(r'^(.+?)([一二三123])([A-Za-z])$', text)
    if m:
        prefix = m.group(1)
        year_str = m.group(2)
        class_str = m.group(3)
        year_num = {'一': 1, '二': 2, '三': 3, '1': 1, '2': 2, '3': 3}.get(year_str, 0)
        return (college_rank, prefix, year_num, class_str)

    return (college_rank, text, 0, "")


def apply_normalized_order(result: pd.DataFrame, col: str, college_order, class_order=None):
    if col not in result.columns:
        return result

    if col == "學院":
        order = college_order
    elif col == "班級":
        if class_order is None:
            unique_classes = result[col].dropna().astype(str).unique()
            class_order = sorted(unique_classes, key=lambda c: parse_class_key(c, college_order))
        order = class_order
    else:
        return result

    result[col] = pd.Categorical(result[col].astype(str), categories=order, ordered=True)
    sort_cols = [col] + [c for c in result.columns if c != col]
    result = result.sort_values(sort_cols)
    return result


def normalize_display_table(df):
    out = df.copy()
    if "count" in out.columns:
        out = out.rename(columns={"count": "人數"})
    if "percent" in out.columns:
        out = out.rename(columns={"percent": "百分比"})
        out["百分比"] = out["百分比"].astype(float).round(2).map(lambda x: f"{x:.2f}%")
    return out

def k(name: str) -> str:
    g = group_col if group_col else "nogroup"
    return f"{question}|{qtype}|{g}|{name}"

# -----------------------------
# UI
# -----------------------------
st.title("115學年度大二學生學習投入分析")

with st.sidebar:
    st.header("分析設定")
    top_n = 12
    top_k_rank = 3

    show_pct = st.checkbox("顯示百分比（%）", value=False)

    # 母體篩選（可不選，留空即全校）
    population_attrs = st.multiselect(
        "母體欄位（可複選）", ["學院", "班級"], default=[]
    )

selected_colleges = []
selected_classes = []

pct_mode = None
if show_pct:
    pct_mode = st.radio(
        "百分比母體",
        ["全體百分比", "分組：組內（各組=100%）"],
        index=1,
    )

    st.divider()
    st.caption("提示：若分組欄位（如『班級』『性別』『學院』）沒出現在清單，可在下方改名或先在前處理新增欄位。")

df = load_excel(FILE_PATH, DEFAULT_SHEET)
df = add_college_column( # 學院標籤
    df,
    class_col="班級",
    out_col="學院",
    prefix_to_college=PREFIX_TO_COLLEGE,
    unknown="未分類",
)

# 學院預設順序，沿用大一
college_order = list(dict.fromkeys(PREFIX_TO_COLLEGE.values()))

# 選擇母體值
if "學院" in population_attrs and "學院" in df.columns:
    with st.sidebar:
        college_values = [x for x in college_order if x in df["學院"].dropna().astype(str).unique()]
        # 若有未歸類也放最後
        extras = [x for x in sorted(df["學院"].dropna().astype(str).unique()) if x not in college_values]
        selected_colleges = st.multiselect(
            "選取學院（可多選）",
            college_values + extras,
            default=[],
        )
if "班級" in population_attrs and "班級" in df.columns:
    with st.sidebar:
        class_values = df["班級"].dropna().astype(str).unique()
        class_ordered = sorted(class_values, key=lambda c: parse_class_key(c, college_order))
        selected_classes = st.multiselect(
            "選取班級（可多選）",
            class_ordered,
            default=[],
        )

# 母體過濾（學院/班級）
mask = pd.Series(True, index=df.index)
if population_attrs:
    if selected_colleges or selected_classes:
        mask = pd.Series(False, index=df.index)
        if selected_colleges:
            mask |= df["學院"].isin(selected_colleges)
        if selected_classes:
            mask |= df["班級"].isin(selected_classes)
    else:
        mask = pd.Series(False, index=df.index)

if not mask.any():
    st.warning("母體篩選後無資料，請調整學院 / 班級選擇。")

df = df[mask]

# infer types
tagger = SurveyColumnTypeTagger()
type_rows = []
for c in df.columns:
    res = tagger.tag_column(df, c)
    type_rows.append({"column": c, "qtype": res.qtype, "confidence": res.confidence, "reason": res.reason})
types_df = pd.DataFrame(type_rows)

# Choose group columns (meta + user choice)
meta_candidates = types_df.loc[types_df["qtype"].isin(["meta"]), "column"].tolist()
group_candidates = meta_candidates.copy()
if "學院" in df.columns and "學院" not in group_candidates:
    group_candidates.insert(0, "學院")

# also allow any column as group (some users put demographics as single_choice)
extra_group_candidates = types_df.loc[types_df["qtype"].isin(["single_choice", "likert"]), "column"].tolist()
for x in extra_group_candidates:
    if x not in group_candidates:
        group_candidates.append(x)

# Filter out question columns for selection
question_cols = types_df.loc[~types_df["qtype"].isin(["meta"]), "column"].tolist()

group_col = st.selectbox("分組欄位（可不選）", ["(不分組)"] + group_candidates, index=0)
if group_col == "(不分組)":
    group_col = None
filtered_questions = types_df[types_df["column"].isin(question_cols)]
question = st.selectbox("選擇題目（欄位）", filtered_questions["column"].tolist())

# show classification
row = types_df[types_df["column"] == question].iloc[0]
#st.info(f"題型判斷：**{row['qtype']}**（信心 {row['confidence']:.2f}）｜{row['reason']}")

# -----------------------------
# Plot by type
# -----------------------------
qtype = row["qtype"]

st.divider()

if qtype in ["single_choice", "likert"]:
    show_all = question in ["學院", "班級"]
    if group_col:
        vc = group_value_counts(df, question, group_col=group_col)

        if not show_all:
            # top-N by total count
            totals = vc.groupby(question)["count"].sum().sort_values(ascending=False).head(top_n).index.tolist()
            vc = vc[vc[question].isin(totals)]

        # 依學院/班級順序排序
        if question in ["學院", "班級"]:
            vc = apply_normalized_order(vc, question, college_order)
        if group_col in ["學院", "班級"]:
            vc = apply_normalized_order(vc, group_col, college_order)

        if show_pct:
            if pct_mode == "全體百分比":
                vc = add_percent(vc, value_col=question)   # 不給 group_col
                y_label = "% (overall)"
            else:
                vc = add_percent(vc, value_col=question, group_col=group_col)
                y_label = "% (within group)"

            y_col = "percent"
        else:
            y_col = "count"
            y_label = "count"

        category_orders = {}
        if question == "學院":
            category_orders[question] = college_order
        elif question == "班級":
            category_orders[question] = sorted(vc[question].dropna().astype(str).unique(), key=lambda c: parse_class_key(c, college_order))
        if group_col == "學院":
            category_orders[group_col] = college_order
        elif group_col == "班級":
            category_orders[group_col] = sorted(vc[group_col].dropna().astype(str).unique(), key=lambda c: parse_class_key(c, college_order))

        fig = px.bar(vc, x=question, y=y_col, color=group_col, barmode="group", category_orders=category_orders if category_orders else None)
        fig.update_yaxes(title=y_label)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(normalize_display_table(vc.sort_values([y_col], ascending=False)), width="stretch")
    else:
        vc = df[question].dropna().value_counts().reset_index()
        vc.columns = ["value", "count"]
        if not show_all:
            vc = vc.head(top_n)

        if question in ["學院", "班級"]:
            if question == "學院":
                order = college_order
            else:
                order = sorted(vc["value"].dropna().astype(str).unique(), key=lambda c: parse_class_key(c, college_order))
            vc["value"] = pd.Categorical(vc["value"].astype(str), categories=order, ordered=True)
            vc = vc.sort_values("value")

        if show_pct:
            vc["percent"] = vc["count"] / vc["count"].sum() * 100
            y_col = "percent"
            y_label = "%"
        else:
            y_col = "count"
            y_label = "count"

        fig = px.bar(vc, x="value", y=y_col, category_orders={"value": order} if question in ["學院", "班級"] else None)
        fig.update_yaxes(title=y_label)
        st.plotly_chart(fig, width="stretch")

        display_vc = vc.rename(columns={"value": question})
        st.dataframe(normalize_display_table(display_vc), width="stretch")

elif qtype == "multi_choice":
    if group_col:
        tmp = df[[group_col, question]].dropna()
        # explode within each group
        rows = []
        for g, sub in tmp.groupby(group_col):
            ex = explode_multi(sub[question])
            if not ex.empty:
                cts = ex.value_counts().reset_index()
                cts.columns = ["option", "count"]
                cts[group_col] = g
                rows.append(cts)
        if rows:
            out = pd.concat(rows, ignore_index=True)
            # top-N by total
            top_opts = out.groupby("option")["count"].sum().sort_values(ascending=False).head(top_n).index.tolist()
            out = out[out["option"].isin(top_opts)]
            if show_pct:
                out = add_percent(out, value_col="option", group_col=group_col)  # 組內%
                y_col = "percent"
                y_label = "% (within group)"
            else:
                y_col = "count"
                y_label = "count"

            fig = px.bar(out, x="option", y=y_col, color=group_col, barmode="group")
            fig.update_yaxes(title=y_label)
            st.plotly_chart(fig, width="stretch", key=f"{question}_{group_col}_multi_group")
            st.dataframe(out.sort_values(y_col, ascending=False), width="stretch")
            st.dataframe(out.sort_values("count", ascending=False), width="stretch")
        else:
            st.warning("此題在目前資料中沒有可分析的複選內容。")
    else:
        ex = explode_multi(df[question])
        vc = ex.value_counts().head(top_n).reset_index()
        vc.columns = ["option", "count"]
        fig = px.bar(vc, x="option", y="count")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(vc, width="stretch")

elif qtype == "multi_choice_ranked":
    st.subheader("排名題分析")
    weights = {1: 3, 2: 2, 3: 1}
    st.caption(f"預設加權（Borda）：第1名=3，第2名=2，第3名=1；最多取前 {top_k_rank} 名")

    # Overall
    rank_table, score_df = ranked_stats(df[question], top_k=top_k_rank, weights=weights)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### ① 第一順位（主因）")
        first = df[question].dropna().astype(str).apply(lambda x: re.split(MULTI_SEP_REGEX, x)[0].strip() if str(x).strip() else np.nan)
        first = first.dropna()
        vc = first.value_counts().head(top_n).reset_index()
        vc.columns = ["option", "count"]

        if show_pct:
            vc["percent"] = vc["count"] / vc["count"].sum() * 100
            y_col = "percent"
            y_label = "%"
        else:
            y_col = "count"
            y_label = "count"

        fig = px.bar(vc, x="option", y=y_col)
        fig.update_yaxes(title=y_label)
        st.plotly_chart(fig, width="stretch", key=k("rank_first_overall"))
        st.dataframe(vc, width="stretch")

    with c2:
        st.markdown("### ② 加權排名（整體重要性）")
        if not score_df.empty:
            top_score = score_df.head(top_n)
            fig = px.bar(top_score, x="option", y="weighted_score")
            st.plotly_chart(
                fig,
                width="stretch",
                key=f"{question}_weighted_rank_chart"
            )
            st.dataframe(top_score, width="stretch")
        else:
            st.warning("此題沒有足夠資料做加權排名。")

    st.markdown("### ③ 名次分布（Rank × Option）")
    if not rank_table.empty:
        # convert to long for stacked bar
        long = rank_table.melt(id_vars=["option"], var_name="rank", value_name="count")
        long["rank"] = long["rank"].astype(int)
        # top-N by weighted score
        top_opts = score_df.head(top_n)["option"].tolist() if not score_df.empty else long["option"].unique().tolist()[:top_n]
        long = long[long["option"].isin(top_opts)]
        fig = px.bar(long, x="option", y="count", color="rank", barmode="stack")
        st.plotly_chart(fig, width="stretch", key=k("rank_dist"))
        st.dataframe(rank_table, width="stretch")
    else:
        st.info("目前無法建立名次分布表（可能資料格式不是用 ; 代表排序）。")

    # Grouped ranking (optional)
    if group_col:
        st.markdown(f"### ④ 分組比較（{group_col}）")
        tmp = df[[group_col, question]].dropna()
        groups = tmp[group_col].dropna().astype(str).unique().tolist()
        groups = sorted(groups)[:30]  # 避免類別太多爆炸

        mode = st.radio("分組呈現方式", ["第一順位比例", "加權排名分數"], horizontal=True)

        if mode == "第一順位比例":
            rows = []
            for g in groups:
                sub = tmp[tmp[group_col].astype(str) == str(g)]
                first = sub[question].astype(str).apply(lambda x: re.split(MULTI_SEP_REGEX, x)[0].strip() if str(x).strip() else np.nan).dropna()
                vc = first.value_counts().head(top_n).reset_index()
                vc.columns = ["option", "count"]
                vc[group_col] = g
                rows.append(vc)
            out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            if out.empty:
                st.warning("此分組下沒有足夠資料。")
            else:
                # top-N overall
                top_opts = out.groupby("option")["count"].sum().sort_values(ascending=False).head(top_n).index.tolist()
                out = out[out["option"].isin(top_opts)]
                if show_pct:
                    out["percent"] = out["count"] / out.groupby(group_col)["count"].transform("sum") * 100
                    y_col = "percent"
                    y_label = "% (within group)"
                else:
                    y_col = "count"
                    y_label = "count"
                fig = px.bar(out, x="option", y=y_col, color=group_col, barmode="group")
                fig.update_yaxes(title=y_label)
                st.plotly_chart(fig, width="stretch", key=k("rank_first_group"))
                st.dataframe(out.sort_values("count", ascending=False), width="stretch")

        else:  # weighted score (use group-wise average)
            rows = []
            for g in groups:
                sub = tmp[tmp[group_col].astype(str) == str(g)]
                score_tbl = ranked_score_table(sub[question], top_k=top_k_rank, weights=weights)
                if not score_tbl.empty:
                    score_tbl[group_col] = g
                    rows.append(score_tbl)

            out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            if out.empty:
                st.warning("此分組下沒有足夠資料。")
            else:
                # 用「平均分」挑 Top-N（組間公平）
                top_opts = (out.groupby("option")["avg_score"]
                            .mean()
                            .sort_values(ascending=False)
                            .head(top_n)
                            .index.tolist())
                out = out[out["option"].isin(top_opts)]

                fig = px.bar(out, x="option", y="avg_score", color=group_col, barmode="group")
                fig.update_yaxes(title="avg weighted score (per respondent)")
                st.plotly_chart(fig, width="stretch", key=k("rank_weighted_overall"))

                # 想同時看樣本數很重要
                st.dataframe(out.sort_values("avg_score", ascending=False), width="stretch")

elif qtype == "short_answer":
    st.subheader("簡答題分析")
    s = df[question].dropna().astype(str)
    if s.empty:
        st.info("此題目前沒有回覆。")
    else:
        # very lightweight "keyword" freq (no jieba to keep deps minimal)
        # you can later replace with jieba if you want better Chinese tokenization
        st.caption("提示：這裡先用簡易字詞統計（未做中文斷詞），若要更準可改用 jieba。")
        text = " ".join(s.tolist())
        # keep CJK chars and letters/numbers
        tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}|\d+", text)
        stop = {"因為", "覺得", "可以", "沒有", "就是", "但是", "以及", "所以", "目前", "自己", "比較"}
        tok = [t for t in tokens if t not in stop]
        top = pd.Series(tok).value_counts().head(top_n).reset_index()
        top.columns = ["token", "count"]
        fig = px.bar(top, x="token", y="count")
        st.plotly_chart(fig, width="stretch")

        # filter + show raw
        keyword = st.text_input("用關鍵字篩選回覆（可空白）", "")
        if keyword.strip():
            view = s[s.str.contains(keyword, na=False)]
        else:
            view = s
        st.write(f"回覆筆數：{len(view)} / {len(s)}")
        st.dataframe(view.to_frame("response").reset_index(drop=True), width="stretch", height=420)

else:
    st.warning("此欄位暫時無法判斷題型或沒有足夠資訊。你可以先看原始分佈：")
    st.dataframe(df[question].dropna().astype(str).value_counts().head(30).reset_index(), width="stretch")

# -----------------------------
# Debug panel: show all columns and inferred types
# -----------------------------
#with st.expander("（除錯）查看所有欄位的題型判斷結果"):
#    st.dataframe(types_df.sort_values(["qtype", "confidence"], ascending=[True, False]), width="stretch")
