"""1주차: 합성 앱 로그 생성기 (iM뱅크 신규 가입 고객 이탈 예측 프로토타입)

실행:  pip install numpy pandas duckdb   (duckdb가 없으면 SQLite 파일로 대신 저장)
       python data/generate_data.py            # 고객 20,000명, data/imbank.duckdb 생성

설계 요약 (발표 때 그대로 공개할 것 - 이 데이터는 '합성'이다)
- 고객마다 숨은 '관심도'(engagement)를 두고, 이 값이 일별 이벤트 발생률을 결정한다.
- 연령대·유입 경로가 관심도 분포에 영향을 준다 (50대 이상 비중 높음, 제휴 유입은 낮은 편).
- 발생률은 가입 후 시간이 지날수록 지수적으로 감소한다 (고객마다 감소 속도가 다름).
- 이탈 정의: 가입 후 8~30일 사이 로그인 1회 이하.  예측 시점: 가입 후 7일.
"""
import argparse
import os

import sqlite3

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:  # duckdb가 없으면 SQLite로 대신 저장 (구조는 동일)
    duckdb = None

EVENT_TYPES = ["login", "balance", "transfer", "card", "push_open"]
# 이벤트별 기본 발생률 비중 (로그인 대비)
EVENT_WEIGHT = {"login": 1.0, "balance": 0.7, "transfer": 0.25, "card": 0.2, "push_open": 0.15}
AGE_GROUPS = ["10-20s", "30s", "40s", "50s+"]
AGE_PROBS = [0.18, 0.17, 0.20, 0.45]  # 이용자 다수가 50대 이상이라는 가정
AGE_SHIFT = {"10-20s": 0.35, "30s": 0.15, "40s": 0.0, "50s+": -0.15}  # 관심도 평균 이동
CHANNEL_SHIFT = {"organic": 0.2, "partner": -0.3}  # 제휴 유입은 이용 정체가 큼
OBS_DAYS = 30  # 관측 기간


def make_customers(n, rng):
    age = rng.choice(AGE_GROUPS, size=n, p=AGE_PROBS)
    channel = rng.choice(["organic", "partner"], size=n, p=[0.6, 0.4])
    signup_offset = rng.integers(0, 90, size=n)  # 가입일 분산
    signup_date = pd.Timestamp("2026-01-01") + pd.to_timedelta(signup_offset, unit="D")

    # 숨은 관심도: 정규분포 + 연령/유입 효과 (평균 0 근처)
    latent = (rng.normal(0, 1.0, n)
              + np.array([AGE_SHIFT[a] for a in age])
              + np.array([CHANNEL_SHIFT[c] for c in channel]))
    # 시간에 따른 감소 속도: 관심 낮을수록 빨리 식음
    decay = np.clip(0.10 - 0.04 * latent + rng.normal(0, 0.02, n), 0.01, 0.35)
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "age_group": age,
        "channel": channel,
        "signup_date": signup_date,
        "latent_engagement": latent,   # 모델 입력에 쓰지 말 것 (정답 누수)
        "decay": decay,
    })


def make_events(customers, rng, base_rate):
    rows = []
    days = np.arange(OBS_DAYS)  # 가입 후 0~29일
    for r in customers.itertuples(index=False):
        # 일별 로그인 기대 횟수: base * exp(0.9*latent) * exp(-decay*day)
        lam_login = base_rate * np.exp(0.9 * r.latent_engagement) * np.exp(-r.decay * days)
        for etype, w in EVENT_WEIGHT.items():
            counts = rng.poisson(lam_login * w)
            for d in days[counts > 0]:
                for _ in range(counts[d]):
                    # 하루 안의 시각은 낮 시간대 위주
                    hour = int(np.clip(rng.normal(14, 5), 0, 23))
                    rows.append((r.customer_id, etype, int(d), hour))
    ev = pd.DataFrame(rows, columns=["customer_id", "event_type", "day_since_signup", "hour"])
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-rate", type=float, default=0.25,
                    help="관심도 0 고객의 첫날 로그인 기대 횟수 (활성화율 약 45%%가 되도록 조정)")
    ap.add_argument("--sqlite", action="store_true", help="duckdb가 있어도 SQLite로 저장 (웹앱은 SQLite를 읽음)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "imbank.duckdb"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cust = make_customers(args.n, rng)
    ev = make_events(cust, rng, args.base_rate)

    # 라벨: 가입 후 8~30일(day 7~29) 로그인 수 -> 1회 이하이면 이탈
    late_login = (ev[(ev.event_type == "login") & (ev.day_since_signup >= 7)]
                  .groupby("customer_id").size())
    cust["late_logins"] = cust.customer_id.map(late_login).fillna(0).astype(int)
    cust["churn"] = (cust.late_logins <= 1).astype(int)

    if duckdb is not None and not args.sqlite:
        con = duckdb.connect(args.out)
        con.execute("DROP TABLE IF EXISTS customers")
        con.execute("DROP TABLE IF EXISTS events")
        con.register("cust_df", cust)
        con.register("ev_df", ev)
        con.execute("CREATE TABLE customers AS SELECT * FROM cust_df")
        con.execute("CREATE TABLE events AS SELECT * FROM ev_df")
        con.close()
    else:
        args.out = args.out.replace(".duckdb", ".sqlite")
        con = sqlite3.connect(args.out)
        cust.assign(signup_date=cust.signup_date.astype(str)).to_sql("customers", con, if_exists="replace", index=False)
        ev.to_sql("events", con, if_exists="replace", index=False)
        con.close()

    # 요약 출력: 목표 활성화율(약 45%)과 맞는지 확인
    active_rate = 1 - cust.churn.mean()
    print(f"고객 수: {len(cust):,} / 이벤트 수: {len(ev):,}")
    print(f"가입 후 8~30일 활성화율(=1-이탈률): {active_rate:.1%}  (목표 약 45%)")
    print(cust.groupby("age_group").churn.mean().round(3).to_string())
    print(cust.groupby("channel").churn.mean().round(3).to_string())
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
