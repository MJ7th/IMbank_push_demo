"""2주차: 가입 7일째 이탈 예측 모델 학습 -> risk_scores 테이블 저장

실행:  python model/train.py            (먼저 python data/generate_data.py --sqlite)
- 입력은 가입 후 0~6일(첫 7일) 행동만 사용 (7일 이후 이벤트·숨은 열은 쓰지 않음: 누수 방지)
- 모델: scikit-learn HistGradientBoosting (LightGBM과 같은 계열). lightgbm이 설치돼 있으면 그것을 사용.
"""
import os
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMClassifier as Model
    MODEL_NAME = "LightGBM"
    def make_model():
        return Model(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=42, verbose=-1)
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier as Model
    MODEL_NAME = "HistGradientBoosting"
    def make_model():
        return Model(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=42)

DB = os.path.join(os.path.dirname(__file__), "..", "data", "imbank.sqlite")
EVENT_TYPES = ["login", "balance", "transfer", "card", "push_open"]
FEATURES = [f"n_{e}" for e in EVENT_TYPES] + ["login_days", "days_since_last_login", "age_code", "channel_partner"]


def build_features(con):
    cust = pd.read_sql("SELECT customer_id, age_group, channel, churn FROM customers", con)
    ev = pd.read_sql("SELECT customer_id, event_type, day_since_signup FROM events WHERE day_since_signup < 7", con)
    cnt = ev.pivot_table(index="customer_id", columns="event_type", values="day_since_signup",
                         aggfunc="count", fill_value=0).reindex(columns=EVENT_TYPES, fill_value=0)
    cnt.columns = [f"n_{c}" for c in cnt.columns]
    login = ev[ev.event_type == "login"].groupby("customer_id").day_since_signup
    feat = cust.set_index("customer_id").join(cnt).fillna(0)
    feat["login_days"] = login.nunique().reindex(feat.index).fillna(0)
    last = login.max().reindex(feat.index)
    feat["days_since_last_login"] = (6 - last).fillna(7)  # 첫 7일 내 로그인이 없으면 7
    feat["age_code"] = feat.age_group.map({"10-20s": 0, "30s": 1, "40s": 2, "50s+": 3})
    feat["channel_partner"] = (feat.channel == "partner").astype(int)
    return feat


def main():
    con = sqlite3.connect(DB)
    feat = build_features(con)
    X, y = feat[FEATURES], feat["churn"]
    # 학습 70 / 검증 15 / 테스트 15 (이탈 비율 유지)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    model = make_model().fit(X_tr, y_tr)
    for name, (Xs, ys) in {"검증": (X_va, y_va), "테스트": (X_te, y_te)}.items():
        p = model.predict_proba(Xs)[:, 1]
        pred = (p >= 0.5).astype(int)
        k = int(len(p) * 0.30)
        top = np.argsort(-p)[:k]
        print(f"[{MODEL_NAME}] {name}: AUC={roc_auc_score(ys, p):.3f} "
              f"재현율={recall_score(ys, pred):.3f} 정밀도={precision_score(ys, pred):.3f} "
              f"위험 상위 30%가 잡은 이탈 고객={ys.values[top].sum() / ys.sum():.1%}")

    risk = pd.DataFrame({"customer_id": feat.index, "risk": model.predict_proba(X)[:, 1]})
    split = pd.Series("train", index=feat.index)
    split[X_va.index] = "valid"
    split[X_te.index] = "test"
    risk["split"] = split.values
    risk.to_sql("risk_scores", con, if_exists="replace", index=False)
    con.close()
    print(f"risk_scores 저장 완료 ({len(risk):,}명)")


if __name__ == "__main__":
    main()
