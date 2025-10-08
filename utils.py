# utils.py
import re
import pandas as pd
from typing import List, Dict, Any
import os
import csv

SKILL_SEP_RE = re.compile(r'[;,|\\/]+|\band\b|\n', flags=re.IGNORECASE)

def parse_skills_field(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(s).strip().lower() for s in x if str(s).strip()]
    s = str(x)
    if not s or s.lower() in ("nan", "none", "null"):
        return []
    parts = [p.strip().lower() for p in SKILL_SEP_RE.split(s) if p.strip()]
    return parts

def safe_read_csv(path: str) -> pd.DataFrame:
    """
    Try multiple delimiters if normal read fails.
    Attempt to detect tab/semicolon/pipe etc.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # quick sniff
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        for sep in [',', '\t', ';', '|']:
            try:
                df = pd.read_csv(path, sep=sep, engine='python', quoting=csv.QUOTE_MINIMAL)
                # if the file loads with >1 col, accept
                if df.shape[1] > 1 or sep == '\t':
                    return df
            except Exception:
                continue
    # last attempt: read with python engine no sep and then split columns heuristically
    df = pd.read_csv(path, engine='python', error_bad_lines=False)
    return df

def combine_skill_columns(df: pd.DataFrame, candidate_names: List[str]=None) -> pd.Series:
    """
    Given a dataframe, try to produce a single series of skills (as lists).
    Looks for columns with 'skill' in name or numeric binary columns etc.
    """
    candidates = candidate_names or [c for c in df.columns if 'skill' in c.lower()]
    if candidates:
        # pick the most popular non-empty column or combine multiple
        combined = df[candidates].fillna('').astype(str).agg(';'.join, axis=1)
        return combined.apply(parse_skills_field)
    # fallback: find binary indicator columns (0/1)
    bin_cols = []
    for c in df.columns:
        vals = df[c].dropna().unique()
        vals_small = set([str(v) for v in vals if len(str(v)) <= 3])
        if vals_small.issubset({'0','1','0.0','1.0','True','False','true','false'}):
            bin_cols.append(c)
    if bin_cols:
        def row_to_skills(r):
            skills = []
            for c in bin_cols:
                v = r.get(c)
                if pd.isna(v):
                    continue
                if str(v).strip() in ('1','1.0','True','true'):
                    skills.append(c.strip().lower())
            return skills
        return df.apply(row_to_skills, axis=1)
    # else create empty lists
    return pd.Series([[] for _ in range(len(df))], index=df.index)

def find_course_skill_map(dfs) -> Dict[str, List[str]]:
    """
    Search list of dataframes for one that maps course->skills and return dict.
    Looks for course-like columns and skill-like columns.
    """
    for df in dfs:
        cols = list(df.columns)
        low = [c.lower() for c in cols]
        course_cols = [c for c in cols if 'course' in c.lower() or 'course_id' in c.lower() or 'course name' in c.lower()]
        skill_cols = [c for c in cols if 'skill' in c.lower() or 'skills' in c.lower() or 'skill_set' in c.lower()]
        if course_cols and skill_cols:
            course_col = course_cols[0]
            skill_col = skill_cols[0]
            mapping = {}
            for _,r in df.iterrows():
                cname = str(r[course_col]).strip()
                skl = parse_skills_field(r[skill_col])
                if cname:
                    mapping[cname] = skl
            if mapping:
                return mapping
    return {}
    