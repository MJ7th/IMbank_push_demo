"""DB의 테스트 고객과 위험 점수를 index.html 안의 데이터로 내보낸다.

실행:  python tools/export_demo.py     (먼저 generate_data.py --sqlite, train.py 실행)
index.html 의  /*DATA_START*/ ... /*DATA_END*/  사이를 새 데이터로 바꾼다.
- 모델이 학습에 쓰지 않은 테스트 집합 고객만 내보낸다 (숨은 열 latent_engagement, decay 는 제외).
"""
import json
import os
import re
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
DB = os.path.join(ROOT, "data", "imbank.sqlite")
HTML = os.path.join(ROOT, "index.html")
AGE = {"10-20s": 0, "30s": 1, "40s": 2, "50s+": 3}
CH = {"organic": 0, "partner": 1}


def main():
    con = sqlite3.connect(DB)
    cu = pd.read_sql("SELECT c.customer_id id, c.age_group, c.channel, c.churn, c.push_consent, r.risk, r.split "
                     "FROM customers c JOIN risk_scores r USING(customer_id)", con)
    l7 = pd.read_sql("SELECT customer_id id, COUNT(*) n7 FROM events "
                     "WHERE event_type='login' AND day_since_signup<7 GROUP BY 1", con)
    n_events = int(con.execute("SELECT COUNT(*) FROM events").fetchone()[0])
    con.close()
    cu = cu.merge(l7, on="id", how="left").fillna({"n7": 0})
    t = cu[cu.split == "test"]
    # [고객ID, 연령대, 유입, 첫 7일 로그인, 위험 점수, 이탈 여부, 수신 동의]
    rows = [[int(r.id), AGE[r.age_group], CH[r.channel], int(r.n7), round(float(r.risk), 3),
             int(r.churn), int(r.push_consent)] for r in t.itertuples()]
    y, p = t.churn.values, t.risk.values
    top = np.argsort(-p)[:int(len(t) * 0.30)]
    metrics = dict(n=len(t), total=len(cu), total_events=n_events,
                   active_rate=round(float(1 - cu.churn.mean()), 3),
                   consent_rate=round(float(cu.push_consent.mean()), 3),
                   auc=round(float(roc_auc_score(y, p)), 3),
                   recall=round(float(recall_score(y, p >= 0.5)), 3),
                   precision=round(float(precision_score(y, p >= 0.5)), 3),
                   top30=round(float(y[top].sum() / y.sum()), 3))
    data = json.dumps(dict(rows=rows, metrics=metrics), separators=(",", ":"))
    html = open(HTML, encoding="utf8").read()
    new, n = re.subn(r"/\*DATA_START\*/.*?/\*DATA_END\*/", lambda m: "/*DATA_START*/" + data + "/*DATA_END*/",
                     html, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("index.html 에서 DATA 표시를 찾지 못했습니다")
    open(HTML, "w", encoding="utf8").write(new)
    print("내보내기 완료:", metrics)


if __name__ == "__main__":
    main()
