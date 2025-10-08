# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import json
import numpy as np
from utils import parse_skills_field
import traceback

MODEL_DIR = os.environ.get("MODEL_DIR", "models")

COURSE_MODEL = os.path.join(MODEL_DIR, "course_clf.joblib")
JOB_MODEL = os.path.join(MODEL_DIR, "job_clf.joblib")
MLB_SKILLS = os.path.join(MODEL_DIR, "mlb_skills.joblib")
TFIDF = os.path.join(MODEL_DIR, "tfidf_asp.joblib")
COURSE_LABELS_JSON = os.path.join(MODEL_DIR, "course_label_names.json")
JOB_LABELS_JSON = os.path.join(MODEL_DIR, "job_label_names.json")

app = Flask(__name__)
CORS(app)

_artifacts = {}

def load_artifacts():
    if _artifacts:
        return _artifacts

    if os.path.exists(COURSE_MODEL):
        _artifacts['course_clf'] = joblib.load(COURSE_MODEL)
    if os.path.exists(JOB_MODEL):
        _artifacts['job_clf'] = joblib.load(JOB_MODEL)
    if os.path.exists(MLB_SKILLS):
        _artifacts['mlb_skills'] = joblib.load(MLB_SKILLS)
    if os.path.exists(TFIDF):
        _artifacts['tfidf'] = joblib.load(TFIDF)

    # Load course labels
    if os.path.exists(os.path.join(MODEL_DIR, "mlb_course.joblib")):
        _artifacts['mlb_course'] = joblib.load(os.path.join(MODEL_DIR, "mlb_course.joblib"))
        _artifacts['course_names'] = _artifacts['mlb_course'].classes_.tolist()
    elif os.path.exists(COURSE_LABELS_JSON):
        with open(COURSE_LABELS_JSON) as f:
            _artifacts['course_names'] = json.load(f)

    # Load job labels
    if os.path.exists(os.path.join(MODEL_DIR, "mlb_job.joblib")):
        _artifacts['mlb_job'] = joblib.load(os.path.join(MODEL_DIR, "mlb_job.joblib"))
        _artifacts['job_names'] = _artifacts['mlb_job'].classes_.tolist()
    elif os.path.exists(JOB_LABELS_JSON):
        with open(JOB_LABELS_JSON) as f:
            _artifacts['job_names'] = json.load(f)

    return _artifacts


def featurize(skills_input, aspiration_input):
    art = load_artifacts()
    skills_list = parse_skills_field(skills_input)

    # skills vector
    skills_vec = art['mlb_skills'].transform([skills_list]) if 'mlb_skills' in art else np.zeros((1, 0))
    # aspiration vector
    asp_vec = art['tfidf'].transform([aspiration_input]).toarray() if 'tfidf' in art else np.zeros((1, 0))

    # combine features
    if skills_vec.size and asp_vec.size:
        X = np.hstack([skills_vec, asp_vec])
    elif skills_vec.size:
        X = skills_vec
    elif asp_vec.size:
        X = asp_vec
    else:
        X = np.zeros((1, 1))

    return X

def predict_topk(clf, X, names, top_k=6):
    """
    Predict probabilities / decision function and return top K labels with scores
    """
    try:
        probs = clf.predict_proba(X)[0]  # first sample
    except Exception:
        probs = clf.decision_function(X)
        if isinstance(probs, np.ndarray) and probs.ndim > 1:
            probs = probs[0]

    topk = sorted(list(enumerate(probs)), key=lambda x: -x[1])[:top_k]
    results = []
    for i, s in topk:
        s_val = float(np.array(s).ravel()[0]) if isinstance(s, (list, tuple, np.ndarray)) else float(s)
        label_name = names[i] if i < len(names) else f"label_{i}"
        results.append({"name": label_name, "score": s_val})
    return results

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json or {}
        skills = data.get("skills") or data.get("skills_text") or data.get("skill_list") or ""
        aspiration = data.get("aspiration") or data.get("aspirations") or ""

        X = featurize(skills, aspiration)
        art = load_artifacts()
        out = {}

        # Courses
        if 'course_clf' in art:
            course_names = art.get('course_names') or []
            top_courses = predict_topk(art['course_clf'], X, course_names)
            out['courses'] = [{"course": c["name"], "score": c["score"]} for c in top_courses]

        # Job roles
        if 'job_clf' in art:
            job_names = art.get('job_names') or []
            top_jobs = predict_topk(art['job_clf'], X, job_names)
            out['job_roles'] = [{"job": j["name"], "score": j["score"]} for j in top_jobs]
        
        # Career pathway
        if 'courses' in out and 'job_roles' in out:
            out['career_pathway'] = {
                "learner_skills": skills,
                "aspiration": aspiration,
                "path": {
                    "courses": out['courses'],
                    "job_roles": out['job_roles']
                }
            }

        return jsonify(out)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
