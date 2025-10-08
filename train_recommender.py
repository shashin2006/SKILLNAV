import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from utils import safe_read_csv, combine_skill_columns, parse_skills_field, find_course_skill_map
import warnings
warnings.filterwarnings("ignore")


def load_all(paths):
    dfs = []
    for p in paths:
        df = safe_read_csv(p)
        df['__source_file'] = os.path.basename(p)
        dfs.append(df)
    return dfs


def build_features(learners_df):
    skills_series = combine_skill_columns(learners_df)
    mlb = MultiLabelBinarizer(sparse_output=False)
    if skills_series.map(len).sum() > 0:
        skill_matrix = mlb.fit_transform(skills_series)
    else:
        skill_matrix = np.zeros((len(learners_df), 0))
        mlb.fit([[]])

    possible = [c for c in learners_df.columns if any(k in c.lower() for k in ('aspir','goal','career','about','summary','objective'))]
    if possible:
        text_col = possible[0]
        texts = learners_df[text_col].fillna("").astype(str).tolist()
    else:
        possible = [c for c in learners_df.columns if any(k in c.lower() for k in ('description','bio','profile'))]
        texts = learners_df[possible[0]].fillna("").astype(str).tolist() if possible else [""]*len(learners_df)

    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    text_matrix = tfidf.fit_transform(texts).toarray() if any(t.strip() for t in texts) else np.zeros((len(learners_df),0))

    if skill_matrix.size and text_matrix.size:
        X = np.hstack([skill_matrix, text_matrix])
    elif skill_matrix.size:
        X = skill_matrix
    elif text_matrix.size:
        X = text_matrix
    else:
        X = np.zeros((len(learners_df),1))
    return X, mlb, tfidf, skills_series


def build_course_labels(learners_df, course_skill_map):
    # check if explicit course column exists
    course_col = None
    for c in learners_df.columns:
        if 'course' in c.lower():
            course_col = c
            break

    if course_col and learners_df[course_col].notna().sum() > 0:
        def parse_multi(x):
            if pd.isna(x): return []
            if isinstance(x, (list,tuple)): return x
            return [s.strip() for s in str(x).split(';') if s.strip()]
        labels = learners_df[course_col].apply(parse_multi)
        mlb_course = MultiLabelBinarizer()
        Y = mlb_course.fit_transform(labels)
        return Y, mlb_course, mlb_course.classes_.tolist()

    # fallback: construct heuristic labels using course_skill_map
    course_names = list(course_skill_map.keys()) if course_skill_map else []
    if not course_names:
        # if nothing, create dummy course labels from skills
        course_names = ["Course_"+str(i) for i in range(1, 11)]
    Y = np.zeros((len(learners_df), len(course_names)), dtype=int)
    learner_skills = combine_skill_columns(learners_df)
    for i, skl in enumerate(learner_skills):
        sset = set(skl)
        for j, cname in enumerate(course_names):
            req = set(course_skill_map.get(cname, [])) if course_skill_map else set()
            if not req and len(sset) > 0:
                Y[i,j] = 1
            else:
                overlap = len(sset & req)
                if overlap >= max(1, len(req)//2):
                    Y[i,j] = 1
    mlb_course = MultiLabelBinarizer()
    mlb_course.fit([course_names])
    return Y, mlb_course, course_names


def build_job_labels(learners_df):
    job_col = None
    for c in learners_df.columns:
        if 'job' in c.lower() or 'role' in c.lower():
            job_col = c
            break

    if job_col and learners_df[job_col].notna().sum() > 0:
        def parse_multi(x):
            if pd.isna(x): return []
            if isinstance(x, (list,tuple)): return x
            return [s.strip() for s in str(x).split(';') if s.strip()]
        labels = learners_df[job_col].apply(parse_multi)
        mlb_job = MultiLabelBinarizer()
        Y = mlb_job.fit_transform(labels)
        return Y, mlb_job, mlb_job.classes_.tolist()

    # fallback dummy job labels
    job_names = ["Job_"+str(i) for i in range(1, 11)]
    Y = np.zeros((len(learners_df), len(job_names)), dtype=int)
    mlb_job = MultiLabelBinarizer()
    mlb_job.fit([job_names])
    return Y, mlb_job, job_names


def train_model(X, Y):
    if Y is None or Y.shape[1] == 0:
        return None, {}
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=2000))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='micro')
    return clf, {"f1_micro": float(f1)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_paths", nargs="+", required=True)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dfs = load_all(args.data_paths)
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    learners = combined.copy()

    course_skill_map = find_course_skill_map(dfs)

    print("Building features...")
    X, mlb_skills, tfidf, skills_series = build_features(learners)
    print(f"Feature shape: {X.shape}")

    print("Building course labels...")
    Y_courses, mlb_course, course_labels_list = build_course_labels(learners, course_skill_map)
    print("Building job labels...")
    Y_jobs, mlb_job, job_labels_list = build_job_labels(learners)

    trained = {}
    if Y_courses is not None:
        print("Training course model...")
        clf_c, stats_c = train_model(X, Y_courses)
        if clf_c:
            joblib.dump(clf_c, os.path.join(args.out_dir, "course_clf.joblib"))
            trained['course'] = stats_c
            joblib.dump(mlb_course, os.path.join(args.out_dir, "mlb_course.joblib"))
            with open(os.path.join(args.out_dir,"course_label_names.json"), "w") as f:
                json.dump(course_labels_list, f, indent=2)
    else:
        print("No course labels produced; skipping course model.")

    if Y_jobs is not None:
        print("Training job model...")
        clf_j, stats_j = train_model(X, Y_jobs)
        if clf_j:
            joblib.dump(clf_j, os.path.join(args.out_dir, "job_clf.joblib"))
            trained['job'] = stats_j
            joblib.dump(mlb_job, os.path.join(args.out_dir, "mlb_job.joblib"))
            with open(os.path.join(args.out_dir,"job_label_names.json"), "w") as f:
                json.dump(job_labels_list, f, indent=2)
    else:
        print("No job labels produced; skipping job model.")

    # save feature artifacts
    joblib.dump(mlb_skills, os.path.join(args.out_dir, "mlb_skills.joblib"))
    joblib.dump(tfidf, os.path.join(args.out_dir, "tfidf_asp.joblib"))

    learners['skills_list'] = skills_series
    learners[['__source_file']].to_csv(os.path.join(args.out_dir, "trained_learners_meta.csv"), index=False)

    print("Training complete. Artifacts saved to", args.out_dir)
    print("Summary:", trained)
