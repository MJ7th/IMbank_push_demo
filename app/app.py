"""푸시 발송 프로토타입 - 서버(Flask) 버전.  index.html 과 같은 기능을 서버 방식으로 구현한다.

실행:  pip install -r requirements.txt
       python app/app.py          ->  http://localhost:5000
       (DB가 없으면 처음 실행할 때 data/generate_data.py 와 model/train.py 를 자동으로 돌려 만든다)

화면
로그인 화면은 하나다. 계정의 역할에 따라 넘어가는 화면이 다르고, 고객은 관리자 화면을 볼 수 없다.
- /               접속 주소. 로그인 전이면 로그인 · 회원가입 화면(/login)으로 간다.
                  관리자 계정으로 로그인하면 /admin, 고객 계정으로 로그인하면 /app 으로 넘어간다.
- /app            고객 화면: 홈, 알림함, 혜택 안내, 설정(수신 동의, 큰 글씨, 로그아웃). 2초마다 새 알림을 확인한다.
- /admin          관리자 화면: 대상 선택 -> 문구 작성 -> 푸시 발송(대조군 분할) -> 캠페인 현황. 오른쪽에 고객 화면 미리보기.
- /admin/results  관리자 화면: A/B 결과 (가정한 효과를 넣은 시뮬레이션 + 비율 차이 검정 + 민감도 표)
계정: 관리자 admin / demo1234 (시연용). 고객은 회원가입한 계정, 또는 시연용 고객 번호 / 0000.
회원가입: 새 계정에 시연용 합성 고객 한 명(아직 연결되지 않은 고객 중 이탈 위험이 가장 높은 고객)을 연결한다.

로그인과 권한
- 비밀번호는 원문이 아니라 단방향 해시로만 DB에 저장하고, 로그인하면 서버가 서명한 세션 쿠키로 사용자를 구분한다 (30분 뒤 만료).
- 관리자 화면과 관리자 API는 관리자만, 고객 API는 본인(또는 미리보기 중인 관리자)만 쓸 수 있다.
- 로그인을 5회 틀리면 30초 동안 잠긴다.
- 회원가입한 계정은 각자의 비밀번호를 쓴다. 고객 번호로 로그인하는 시연용 계정은 모두 같은 비밀번호(0000)를 쓴다.
  실제 서비스라면 본인 인증과 HTTPS가 필요하다.

index.html 과 다른 점: 발송 기록 · 열람 기록 · 수신 동의가 DB(push_log, campaigns, customers)에 저장되어
새로고침해도 남고, 다른 기기(같은 Wi-Fi의 스마트폰에서 http://<PC의 IP>:5000 접속 후 고객으로 로그인)와 공유된다.

※ iM뱅크 적용을 가정한 시연용이다. 실제 은행 푸시(FCM 등)와 예금 · 적금 · 이체 등 금융 거래 기능은 없다.
   고객 · 이벤트는 합성 데이터, 푸시 효과와 수신 동의 비율은 가정값이다.
"""
import math
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import time
from datetime import timedelta
from functools import wraps

import numpy as np
from flask import Flask, abort, jsonify, redirect, request, session
from scipy.stats import norm
from werkzeug.security import check_password_hash, generate_password_hash

DB = os.path.join(os.path.dirname(__file__), "..", "data", "imbank.sqlite")
BRAND = "데모뱅크"   # 앱 이름은 이 한 곳에서 바꾼다 (실제 은행 이름 · 로고는 쓰지 않는다)
ONLY_TEST = True     # True: 모델이 학습에 쓰지 않은 테스트 고객만 발송 대상으로 쓴다 (index.html 과 같은 고객)
# 시연용 계정 (발표에서 공개하는 값). 실제로 쓰는 비밀번호를 넣지 말 것.
DEMO_ADMIN_ID, DEMO_ADMIN_PW, DEMO_CUSTOMER_PW = "admin", "demo1234", "0000"
MAX_FAILS, LOCK_SECONDS = 5, 30

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET") or secrets.token_hex(32)   # 세션 쿠키 서명 키 (서버를 다시 켜면 모두 로그아웃됨)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  PERMANENT_SESSION_LIFETIME=timedelta(minutes=30))
login_fails = {}   # {아이디: [틀린 횟수, 잠금 해제 시각]}  서버 메모리에만 둔다
signup_times = {}  # {접속 주소: [최근 가입 시각들]}  짧은 시간에 너무 많이 가입하지 못하게 한다
ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{3,19}$")   # 아이디: 영문으로 시작, 4~20자
AGES = ("10-20s", "30s", "40s", "50s+")

LANDS = {"event": "첫 달 이벤트", "report": "소비 리포트", "branch": "영업점 · 앱 함께 쓰기"}
TEMPLATES = {
    "50s+": ("branch", "고객님, 가까운 영업점과 앱을 함께 쓰시면 더 편리합니다. 앱 사용이 어려우시면 영업점에서 도와드립니다. (시연용 문구)"),
    "10-20s": ("report", "가입 첫 달! 오늘 소비 리포트를 확인하고 용돈을 한눈에 관리해 보세요. (시연용 문구)"),
    "": ("event", "첫 달 이벤트가 준비되어 있습니다. 필요한 서비스를 확인해 보세요. (시연용 문구)"),
}


# ------------------------------------------------------------------ DB
def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def ensure_data():
    """DB(고객 · 이벤트 · 위험 점수)가 없으면 1 · 2주차 코드를 실행해 만든다."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    tables = set()
    if os.path.exists(DB):
        con = sqlite3.connect(DB)
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        con.close()
    if not {"customers", "events"} <= tables:
        print("합성 데이터를 만드는 중...")
        subprocess.run([sys.executable, os.path.join(root, "data", "generate_data.py"), "--sqlite"], check=True)
    if "risk_scores" not in tables:
        print("이탈 예측 모델을 학습하는 중...")
        subprocess.run([sys.executable, os.path.join(root, "model", "train.py")], check=True)


def init():
    """앱이 쓰는 표를 준비한다. (customers, risk_scores 는 1 · 2주차 코드가 만든다)"""
    ensure_data()
    con = db()
    con.execute("""CREATE TABLE IF NOT EXISTS campaigns(
        id INTEGER PRIMARY KEY, title TEXT, body TEXT, land TEXT, excluded INTEGER, sent_at REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS push_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT, campaign_id INTEGER, customer_id INTEGER,
        grp TEXT, title TEXT, body TEXT, sent_at REAL, land TEXT, opened_at REAL)""")
    cols = [r["name"] for r in con.execute("PRAGMA table_info(push_log)")]
    for c, t in (("land", "TEXT"), ("opened_at", "REAL")):      # 이전 버전 DB 호환
        if c not in cols:
            con.execute(f"ALTER TABLE push_log ADD COLUMN {c} {t}")
    if "push_consent" not in [r["name"] for r in con.execute("PRAGMA table_info(customers)")]:
        con.execute("ALTER TABLE customers ADD COLUMN push_consent INTEGER DEFAULT 1")  # 이전 버전 데이터: 모두 동의로 간주
    # 계정 표: 비밀번호 원문이 아니라 해시만 저장한다. customer_id 는 회원가입한 계정에 연결된 합성 고객 번호다.
    # 'customer' 행은 고객 번호로 로그인하는 시연용 계정들이 함께 쓰는 비밀번호다.
    con.execute("CREATE TABLE IF NOT EXISTS accounts(login_id TEXT PRIMARY KEY, role TEXT, pw_hash TEXT, customer_id INTEGER)")
    if "customer_id" not in [r["name"] for r in con.execute("PRAGMA table_info(accounts)")]:   # 이전 버전 DB 호환
        con.execute("ALTER TABLE accounts ADD COLUMN customer_id INTEGER")
    for login_id, role, pw in ((DEMO_ADMIN_ID, "admin", DEMO_ADMIN_PW), ("customer", "customer", DEMO_CUSTOMER_PW)):
        if not con.execute("SELECT 1 FROM accounts WHERE login_id=?", (login_id,)).fetchone():
            con.execute("INSERT INTO accounts(login_id, role, pw_hash) VALUES(?,?,?)", (login_id, role, generate_password_hash(pw)))
    con.commit()
    con.close()


def ranked_customers(con, age, channel, top):
    """조건에 맞는 고객을 위험 점수 순으로 정렬해 상위 top% 를 돌려준다. (수신 동의 여부 포함)"""
    q = ("SELECT c.customer_id, c.age_group, c.channel, c.push_consent, r.risk, a.login_id account FROM customers c "
         "JOIN risk_scores r USING(customer_id) LEFT JOIN accounts a ON a.customer_id=c.customer_id WHERE 1=1")
    args = []
    if ONLY_TEST:
        q += " AND r.split='test'"
    if age:
        q += " AND c.age_group=?"; args.append(age)
    if channel:
        q += " AND c.channel=?"; args.append(channel)
    pool = [dict(r) for r in con.execute(q + " ORDER BY r.risk DESC, c.customer_id", args)]
    top = max(1.0, min(100.0, top))
    return pool[:max(1, int(len(pool) * top / 100))] if pool else [], len(pool)


# ------------------------------------------------------------------ 로그인 · 권한
def admin_required(f):
    """관리자로 로그인한 경우에만 실행한다. (화면은 로그인으로 보내고, API는 401을 돌려준다)"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get("role") != "admin":
            return (jsonify(error="관리자 로그인이 필요합니다"), 401) if request.path.startswith("/api/") else redirect("/")   # "/" 가 역할에 맞는 화면(고객 화면 또는 로그인)으로 보낸다
        return f(*args, **kwargs)
    return wrapper


def can_view(cid):
    """고객 화면 · 고객 API: 본인이거나, 미리보기 중인 관리자만 볼 수 있다."""
    return session.get("role") == "admin" or (session.get("role") == "customer" and session.get("cid") == cid)


def start_session(role, cid, name):
    """로그인 성공: 세션에 역할을 기록하고, 역할에 맞는 화면 주소를 돌려준다."""
    session.clear()
    session.permanent = True
    session["role"], session["name"] = role, name
    if role == "customer":
        session["cid"] = cid
    return jsonify(next="/admin" if role == "admin" else "/app")


@app.post("/api/login")
def api_login():
    """로그인 화면은 하나다. 아이디가 어떤 계정인지 서버가 확인해 관리자 화면 또는 고객 화면으로 보낸다."""
    d = request.get_json() or {}
    login_id, pw = str(d.get("id", "")).strip(), str(d.get("pw", ""))
    key = login_id.lower()
    fails = login_fails.setdefault(key, [0, 0.0])
    if fails[1] > time.time():
        return jsonify(error=f"로그인 시도가 많아 잠시 잠겼습니다. {math.ceil(fails[1] - time.time())}초 뒤에 다시 시도하세요."), 429
    con = db()
    role = cid = None
    name = login_id
    acct = con.execute("SELECT login_id, role, pw_hash, customer_id FROM accounts "
                       "WHERE login_id=? COLLATE NOCASE AND login_id!='customer'", (login_id,)).fetchone()
    shared = con.execute("SELECT pw_hash FROM accounts WHERE login_id='customer'").fetchone()["pw_hash"]
    if acct:                                   # 관리자 계정 또는 회원가입한 고객 계정
        if check_password_hash(acct["pw_hash"], pw):
            role, cid, name = acct["role"], acct["customer_id"], acct["login_id"]
    else:                                      # 시연용 고객 번호 계정 (공용 비밀번호). 없는 아이디도 같은 계산을 거친다.
        exists = login_id.isdigit() and con.execute("SELECT 1 FROM customers WHERE customer_id=?", (int(login_id),)).fetchone()
        if check_password_hash(shared, pw) and exists:
            role, cid, name = "customer", int(login_id), f"고객 {int(login_id)}"
    con.close()
    if role is None:
        fails[0] += 1
        if fails[0] >= MAX_FAILS:
            fails[0], fails[1] = 0, time.time() + LOCK_SECONDS
            return jsonify(error=f"{MAX_FAILS}회 틀려 {LOCK_SECONDS}초 동안 로그인이 잠깁니다."), 429
        return jsonify(error=f"아이디 또는 비밀번호가 맞지 않습니다. ({fails[0]}/{MAX_FAILS})"), 401   # 어느 쪽이 틀렸는지는 알려 주지 않는다
    login_fails.pop(key, None)
    return start_session(role, cid, name)


@app.post("/api/signup")
def api_signup():
    """회원가입: 고객 계정을 만들고 시연용 합성 고객 한 명을 연결한 뒤 바로 로그인한다. (관리자 계정은 가입으로 만들 수 없다)"""
    d = request.get_json() or {}
    login_id, pw, age = str(d.get("id", "")).strip(), str(d.get("pw", "")), str(d.get("age", ""))
    if not ID_RE.match(login_id):
        return jsonify(error="아이디는 영문으로 시작하는 4~20자(영문, 숫자, _)로 입력하세요."), 400
    if len(pw) < 6 or len(pw) > 64:
        return jsonify(error="비밀번호는 6자 이상으로 입력하세요."), 400
    if age and age not in AGES:
        return jsonify(error="연령대를 다시 선택하세요."), 400
    recent = [t for t in signup_times.get(request.remote_addr, []) if t > time.time() - 60]
    if len(recent) >= 5:
        return jsonify(error="가입 요청이 많습니다. 잠시 뒤에 다시 시도하세요."), 429
    con = db()
    if con.execute("SELECT 1 FROM accounts WHERE login_id=? COLLATE NOCASE", (login_id,)).fetchone():
        con.close()
        return jsonify(error="이미 사용 중인 아이디입니다."), 409
    # 시연용: 아직 계정에 연결되지 않은 합성 고객 중 이탈 위험이 가장 높은 고객(연령대를 골랐으면 그 연령대)을 연결한다.
    q = ("SELECT c.customer_id FROM customers c JOIN risk_scores r USING(customer_id) "
         "WHERE c.customer_id NOT IN (SELECT customer_id FROM accounts WHERE customer_id IS NOT NULL)")
    args = []
    if ONLY_TEST:
        q += " AND r.split='test'"
    if age:
        q += " AND c.age_group=?"; args.append(age)
    row = con.execute(q + " ORDER BY r.risk DESC, c.customer_id LIMIT 1", args).fetchone()
    if not row:
        con.close()
        return jsonify(error="연결할 수 있는 시연 고객이 없습니다."), 409
    cid = row["customer_id"]
    con.execute("INSERT INTO accounts(login_id, role, pw_hash, customer_id) VALUES(?,?,?,?)",
                (login_id, "customer", generate_password_hash(pw), cid))
    con.execute("UPDATE customers SET push_consent=? WHERE customer_id=?", (1 if d.get("consent") else 0, cid))
    con.commit()
    con.close()
    signup_times[request.remote_addr] = recent + [time.time()]
    return start_session("customer", cid, login_id)


@app.post("/api/logout")
def api_logout():
    session.clear()
    return jsonify(ok=True)


# ------------------------------------------------------------------ 관리자 API
@app.get("/api/targets")
@admin_required
def api_targets():
    age, channel = request.args.get("age", ""), request.args.get("channel", "")
    con = db()
    ranked, pool = ranked_customers(con, age, channel, float(request.args.get("top", 30)))
    con.close()
    excluded = sum(1 for r in ranked if not r["push_consent"])
    land, body = TEMPLATES.get(age, TEMPLATES[""])
    return jsonify(pool=pool, ranked=len(ranked), excluded=excluded, total=len(ranked) - excluded,
                   sample=ranked[:8], land=land, body=body)


@app.post("/api/send")
@admin_required
def api_send():
    d = request.get_json()
    title, body = (d.get("title") or BRAND).strip(), (d.get("body") or "").strip()
    land = d.get("land") if d.get("land") in LANDS else "event"
    if not body:
        return jsonify(error="메시지 내용을 입력하세요"), 400
    con = db()
    ranked, _ = ranked_customers(con, d.get("age", ""), d.get("channel", ""), float(d.get("top", 30)))
    targets = [r for r in ranked if r["push_consent"]]        # 수신 미동의 고객은 발송에서 제외
    if not targets:
        con.close()
        return jsonify(error="발송 대상 고객이 없습니다"), 400
    campaign = con.execute("SELECT COALESCE(MAX(id),0)+1 FROM campaigns").fetchone()[0]
    # 무작위 배정: 순서를 섞은 뒤 앞의 절반을 푸시 받는 군, 나머지를 대조군으로 둔다 (캠페인 번호가 시드 -> 재현 가능)
    order = np.random.default_rng(campaign).permutation(len(targets))
    n_treat = math.ceil(len(targets) / 2) if d.get("control", True) else len(targets)
    treated = set(order[:n_treat].tolist())
    now = time.time()
    con.execute("INSERT INTO campaigns VALUES(?,?,?,?,?,?)",
                (campaign, title, body, land, len(ranked) - len(targets), now))
    con.executemany(
        "INSERT INTO push_log(campaign_id,customer_id,grp,title,body,sent_at,land) VALUES(?,?,?,?,?,?,?)",
        [(campaign, r["customer_id"], "treatment" if i in treated else "control", title, body, now, land)
         for i, r in enumerate(targets)])
    con.commit()
    con.close()
    t_ids = [r["customer_id"] for i, r in enumerate(targets) if i in treated]
    c_ids = [r["customer_id"] for i, r in enumerate(targets) if i not in treated]
    return jsonify(campaign=campaign, sent=len(t_ids), control=len(c_ids), excluded=len(ranked) - len(targets),
                   examples=t_ids[:6], control_examples=c_ids[:2])


@app.get("/api/campaigns")
@admin_required
def api_campaigns():
    con = db()
    rows = [dict(r) for r in con.execute("""
        SELECT c.id, c.land, c.excluded,
               SUM(p.grp='treatment') sent, SUM(p.grp='control') control,
               SUM(p.opened_at IS NOT NULL) opens
        FROM campaigns c JOIN push_log p ON p.campaign_id=c.id GROUP BY c.id ORDER BY c.id""")]
    con.close()
    for r in rows:
        r["land_name"] = LANDS.get(r["land"], r["land"])
    return jsonify(rows)


@app.post("/api/reset")
@admin_required
def api_reset():
    con = db()
    con.execute("DELETE FROM push_log")
    con.execute("DELETE FROM campaigns")
    con.commit()
    con.close()
    return jsonify(ok=True)


def analyze(rows, cid, effect):
    """가정한 효과를 적용해 두 군의 활성 비율을 비교한다. rows: (customer_id, grp, churn)"""
    # 고객마다 0~1 사이의 고정된 난수를 주고, 그 값이 effect 보다 작으면 '푸시로 활성화됨'으로 본다.
    u = np.random.default_rng(int(cid) * 7919).random(len(rows))
    a1 = n1 = a0 = n0 = 0
    for r, ui in zip(rows, u):
        active = 1 - r["churn"]
        if r["grp"] == "treatment":
            if r["churn"] == 1 and ui < effect:
                active = 1
            a1 += active; n1 += 1
        else:
            a0 += active; n0 += 1
    out = dict(n_treat=n1, n_ctrl=n0)
    if n1 and n0:
        p1, p0 = a1 / n1, a0 / n0
        pp = (a1 + a0) / (n1 + n0)
        se0 = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))          # 검정용 (두 군이 같다고 가정)
        se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)      # 신뢰구간용
        z = (p1 - p0) / se0 if se0 else 0
        out.update(rate_treat=p1 * 100, rate_ctrl=p0 * 100, diff=(p1 - p0) * 100,
                   ci_low=(p1 - p0 - 1.96 * se) * 100, ci_high=(p1 - p0 + 1.96 * se) * 100,
                   p_value=float(2 * (1 - norm.cdf(abs(z)))))
    return out


@app.get("/api/results")
@admin_required
def api_results():
    effect = max(0.0, min(100.0, float(request.args.get("effect") or 8))) / 100
    con = db()
    cid = request.args.get("campaign") or con.execute("SELECT MAX(id) FROM campaigns").fetchone()[0]
    if cid is None:
        con.close()
        return jsonify(error="아직 발송한 캠페인이 없습니다")
    rows = con.execute("SELECT p.customer_id, p.grp, c.churn FROM push_log p JOIN customers c USING(customer_id) "
                       "WHERE p.campaign_id=? ORDER BY p.customer_id", (cid,)).fetchall()
    opens = con.execute("SELECT COUNT(*) FROM push_log WHERE campaign_id=? AND opened_at IS NOT NULL", (cid,)).fetchone()[0]
    con.close()
    if not rows:
        return jsonify(error="해당 캠페인이 없습니다")
    out = analyze(rows, cid, effect)
    out.update(campaign=int(cid), effect=effect * 100, opens=opens)
    if "diff" in out:
        out["sens"] = [dict(effect=e, **{k: v for k, v in analyze(rows, cid, e / 100).items() if k in ("diff", "p_value")})
                       for e in (0, 3, 5, 8, 10, 15)]
    return jsonify(out)


# ------------------------------------------------------------------ 고객 API (위험 점수 · 실험 군 정보는 주지 않는다)
@app.get("/api/customer/<int:cid>")
def api_customer(cid):
    if not can_view(cid):
        abort(403)
    con = db()
    r = con.execute("SELECT customer_id, age_group, channel, push_consent FROM customers WHERE customer_id=?", (cid,)).fetchone()
    acct = con.execute("SELECT login_id FROM accounts WHERE customer_id=?", (cid,)).fetchone()
    con.close()
    if not r:
        abort(404)
    out = dict(r)
    out["name"] = acct["login_id"] if acct else f"고객 {cid}"     # 화면에 보여 줄 이름
    return jsonify(out)


@app.get("/api/inbox/<int:cid>")
def api_inbox(cid):
    if not can_view(cid):
        abort(403)
    con = db()
    rows = [dict(r) for r in con.execute(
        "SELECT id,title,body,land,sent_at,opened_at FROM push_log WHERE customer_id=? AND grp='treatment' ORDER BY id DESC",
        (cid,))]
    con.close()
    return jsonify(rows)


@app.post("/api/open/<int:msg_id>")
def api_open(msg_id):
    """고객이 알림을 누르면 열람 시각을 기록한다 (처음 한 번만)."""
    con = db()
    row = con.execute("SELECT customer_id FROM push_log WHERE id=?", (msg_id,)).fetchone()
    if not row or not can_view(row["customer_id"]):      # 자기 알림만 열람 처리할 수 있다
        con.close()
        abort(403)
    con.execute("UPDATE push_log SET opened_at=? WHERE id=? AND grp='treatment' AND opened_at IS NULL", (time.time(), msg_id))
    con.commit()
    con.close()
    return jsonify(ok=True)


@app.post("/api/consent/<int:cid>")
def api_consent(cid):
    if not can_view(cid):
        abort(403)
    v = 1 if request.get_json().get("consent") else 0
    con = db()
    con.execute("UPDATE customers SET push_consent=? WHERE customer_id=?", (v, cid))
    con.commit()
    con.close()
    return jsonify(consent=v)


# ------------------------------------------------------------------ 화면 (HTML)
CSS = """
*{box-sizing:border-box}
body{font-family:'Noto Sans KR','Apple SD Gothic Neo','Malgun Gothic','Noto Sans CJK KR',sans-serif;margin:0;background:#f4f6f9;color:#1b2433;line-height:1.5}
button{font:inherit;cursor:pointer}
header{background:#14264a;color:#fff;padding:11px 20px;display:flex;gap:20px;align-items:center;flex-wrap:wrap}
header b{font-size:17px}header a{color:#cfd9ec;text-decoration:none}header a.on{color:#fff;font-weight:700;text-decoration:underline}
.warn{background:#fff6ee;border-bottom:1px solid #f0cfae;color:#7a3b00;padding:7px 20px;font-size:13.5px}
main{max-width:1220px;margin:16px auto;padding:0 16px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 400px;gap:18px;align-items:start}
@media(max-width:920px){.grid{grid-template-columns:minmax(0,1fr)}}
.card{background:#fff;border:1px solid #d9dee6;border-radius:12px;padding:15px 17px;margin-bottom:13px}
.card h3{margin:0 0 9px;font-size:16.5px;color:#14264a}
label{font-size:13px;color:#5b6572;display:block;margin-bottom:4px}
select,input[type=number],input[type=text],input[type=password],textarea{font:inherit;padding:8px 10px;border:1px solid #b9c2d0;border-radius:8px;width:100%;background:#fff}
.who{margin-left:auto;font-size:14px;display:flex;gap:10px;align-items:center}.who button{background:#ffffff26;color:#fff;border:0;border-radius:8px;padding:5px 11px}
.err{color:#b00020;font-size:13.5px;min-height:1.5em;margin:6px 0}.hint{background:#f3f6fb;border:1px dashed #9fb0cc;border-radius:10px;padding:9px 12px;font-size:13px;color:#33415c}
.row{display:flex;gap:10px;flex-wrap:wrap}.row>div{flex:1;min-width:140px}
.btn{background:#c25a00;color:#fff;border:0;padding:10px 19px;border-radius:8px;font-weight:700}.btn.gray{background:#667}
table{border-collapse:collapse;width:100%;font-size:14px}th{background:#dce6f2;text-align:left}th,td{border:1px solid #c9d1dd;padding:5px 8px}
.note{font-size:13px;color:#5b6572}.ok{background:#e9f7ee;border:1px solid #9bd3ac;padding:9px 12px;border-radius:8px;margin-top:10px}
.big{font-size:29px;font-weight:700;color:#14264a}.sig{color:#0a7a35;font-weight:700}.nsig{color:#a33;font-weight:700}
.bar{display:flex;align-items:center;gap:10px;font-size:14px;margin:6px 0}.bar .l{width:110px}.bar .t{flex:1;background:#e7ebf1;border-radius:6px;height:22px}
.bar .f{height:100%;border-radius:6px;background:#14264a}.bar .f.c{background:#8a97ab}.bar .v{width:60px;text-align:right;font-weight:700}
iframe{width:100%;height:720px;border:0}
"""
WARN = ("<div class=warn>※ iM뱅크 적용을 <b>가정한 시연용 프로토타입</b>(서버 버전)입니다. 실제 iM뱅크 앱이 아니며, 고객 · 이벤트는 "
        "<b>합성 데이터</b>, 푸시 효과는 <b>가정값</b>입니다. 예금 · 적금 · 이체 등 금융 거래 기능은 없으며, 로그인은 <b>시연용 계정</b>으로만 동작합니다.</div>")
LOGOUT_JS = "<script>async function logout(){await fetch('/api/logout',{method:'POST'});location.href='/login'}</script>"


def nav(cur):
    return ("<header><b>푸시 발송 프로토타입</b>"
            f"<a href='/admin' class='{'on' if cur == 'send' else ''}'>① 푸시 발송</a>"
            f"<a href='/admin/results' class='{'on' if cur == 'res' else ''}'>② A/B 결과</a>"
            "<div class=who>관리자 로그인 중 <button onclick='logout()'>로그아웃</button></div></header>" + WARN + LOGOUT_JS)


HEAD = "<!doctype html><html lang=ko><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"

ADMIN = HEAD + "<title>푸시 발송 (관리자)</title><style>" + CSS + "</style>" + nav("send") + """
<main><div class=grid><div>
<div class=card><h3>1. 대상 고객 선택</h3>
<div class=row>
<div><label>연령대</label><select id=age><option value=''>전체</option><option value='10-20s'>10~20대</option><option value='30s'>30대</option><option value='40s'>40대</option><option value='50s+'>50대 이상</option></select></div>
<div><label>유입 경로</label><select id=channel><option value=''>전체</option><option value='organic'>자연 유입</option><option value='partner'>제휴 유입</option></select></div>
<div><label>이탈 위험 상위 (%)</label><input id=top type=number value=30 min=1 max=100></div>
</div>
<p id=count class=note></p>
<table><thead><tr><th>고객 ID</th><th>연령대</th><th>유입</th><th>이탈 위험 점수</th><th>수신 동의</th><th>가입 계정</th></tr></thead><tbody id=tb></tbody></table>
<p class=note>이탈 위험 점수는 가입 후 첫 7일 행동으로 예측한 확률입니다. 수신에 동의하지 않은 고객은 발송에서 제외됩니다.</p></div>

<div class=card><h3>2. 푸시 문구</h3>
<div class=row><div><label>제목</label><input id=title type=text value='__BRAND__'></div>
<div><label>누르면 이동할 화면</label><select id=land><option value=event>첫 달 이벤트</option><option value=report>소비 리포트</option><option value=branch>영업점 · 앱 함께 쓰기</option></select></div></div>
<div style='height:10px'></div>
<label>내용 (연령대를 고르면 시연용 문구가 채워집니다)</label><textarea id=body rows=3></textarea><div style='height:10px'></div>
<label style='display:flex;gap:8px;align-items:center;color:#1b2433'><input type=checkbox id=ctrl checked style='width:auto'> 대조군 분할 (대상의 절반에게는 보내지 않고 효과 비교용으로 남김)</label></div>

<div class=card><h3>3. 발송</h3>
<button class=btn id=send>푸시 발송</button> <button class='btn gray' id=reset>발송 기록 초기화</button>
<div id=result></div></div>

<div class=card><h3>캠페인 현황</h3><div id=camps></div>
<p class=note>열람 수는 고객 화면에서 알림을 실제로 누른 횟수입니다 (DB에 기록, 5초마다 갱신). 활성화 효과는 ② A/B 결과에서 가정값으로 계산합니다.</p></div>
</div>

<div><div class=card><label>고객 화면 미리보기 (관리자는 로그인 없이 고객 화면을 확인합니다)</label><select id=pick></select>
<p class=note style='margin:6px 0 0'>다른 기기에서는 <span id=url></span> 에 접속해 회원가입하거나 시연용 고객 번호로 로그인합니다.</p></div>
<iframe id=frame title='고객 화면'></iframe></div>
</div></main>
<script>
const $=id=>document.getElementById(id),AGE={'10-20s':'10~20대','30s':'30대','40s':'40대','50s+':'50대 이상'},CH={organic:'자연 유입',partner:'제휴 유입'};
let cur=null;
function show(cid){cur=cid;$('frame').src='/app/'+cid;$('url').textContent=location.origin+'/'}
function setPick(opts){$('pick').innerHTML=opts.map(o=>`<option value="${o[0]}">고객 ${o[0]} (${o[1]})</option>`).join('');if(opts.length){if(!opts.find(o=>o[0]==cur))show(opts[0][0]);$('pick').value=cur}}
$('pick').onchange=()=>show(+$('pick').value);
async function load(fill){
  const q=new URLSearchParams({age:$('age').value,channel:$('channel').value,top:$('top').value});
  const d=await (await fetch('/api/targets?'+q)).json();
  $('count').textContent=`위험 상위 ${d.ranked.toLocaleString()}명 (조건에 맞는 고객 ${d.pool.toLocaleString()}명 중) → 수신 미동의 ${d.excluded.toLocaleString()}명 제외 → 발송 대상 ${d.total.toLocaleString()}명`;
  $('tb').innerHTML=d.sample.map(r=>`<tr><td>${r.customer_id}</td><td>${AGE[r.age_group]}</td><td>${CH[r.channel]}</td><td>${r.risk.toFixed(3)}</td><td>${r.push_consent?'동의':'<span class=nsig>미동의 (제외)</span>'}</td><td>${r.account||'-'}</td></tr>`).join('');
  if(fill){$('body').value=d.body;$('land').value=d.land}
  if(cur===null)setPick(d.sample.filter(r=>r.push_consent).slice(0,6).map(r=>[r.customer_id,AGE[r.age_group]]));
}
$('age').addEventListener('change',()=>load(true));['channel','top'].forEach(i=>$(i).addEventListener('change',()=>load(false)));
async function camps(){
  const d=await (await fetch('/api/campaigns')).json();
  $('camps').innerHTML=d.length?'<table><tr><th>캠페인</th><th>발송</th><th>대조군</th><th>수신 미동의 제외</th><th>열람</th><th>이동 화면</th></tr>'+d.map(c=>`<tr><td>#${c.id}</td><td>${c.sent.toLocaleString()}명</td><td>${c.control.toLocaleString()}명</td><td>${c.excluded.toLocaleString()}명</td><td>${c.opens}명</td><td>${c.land_name}</td></tr>`).join('')+'</table>':'<p class=note>아직 발송한 캠페인이 없습니다.</p>';
}
$('send').onclick=async()=>{
  $('send').disabled=true;
  const d=await (await fetch('/api/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
    age:$('age').value,channel:$('channel').value,top:$('top').value,title:$('title').value,body:$('body').value,land:$('land').value,control:$('ctrl').checked})})).json();
  $('send').disabled=false;
  if(d.error){$('result').innerHTML=`<p class=note style='color:#b00'>${d.error}</p>`;return}
  $('result').innerHTML=`<div class=ok>캠페인 #${d.campaign}: 푸시 ${d.sent.toLocaleString()}명 발송, 대조군 ${d.control.toLocaleString()}명, 수신 미동의 제외 ${d.excluded.toLocaleString()}명 — 고객 화면에서 알림을 확인하세요.</div>`;
  const opts=d.examples.map(c=>[c,'푸시 받음']).concat(d.control_examples.map(c=>[c,'대조군 - 받지 않음']));
  if(!d.examples.includes(cur))cur=null;
  setPick(opts);camps();
};
$('reset').onclick=async()=>{await fetch('/api/reset',{method:'POST'});$('result').innerHTML='<div class=ok>발송 기록을 초기화했습니다.</div>';camps();if(cur)show(cur)};
load(true);camps();setInterval(camps,5000);
</script>""".replace("__BRAND__", BRAND)

RESULTS = HEAD + "<title>A/B 결과 (관리자)</title><style>" + CSS + "</style>" + nav("res") + """
<main><div class=card><h3>A/B 실험 결과 (가정 효과 적용 시뮬레이션)</h3>
<div class=row><div><label>캠페인 번호 (비우면 최근)</label><input id=camp type=number min=1></div>
<div><label>가정한 효과: 푸시를 받은 이탈 고객 중 활성으로 돌아서는 비율 (%)</label><input id=eff type=number value=8 min=0 max=100></div></div>
<p class=note>효과는 실제로 측정한 값이 아니라 위에서 입력한 <b>가정값</b>입니다. 활성 = 가입 후 8~30일에 1회 이상 로그인한 고객(MAU 기준, 이탈의 반대).</p></div>
<div id=out></div></main>
<script>
const $=id=>document.getElementById(id),f=x=>x.toFixed(1);
async function load(){
  const d=await (await fetch('/api/results?campaign='+$('camp').value+'&effect='+$('eff').value)).json();
  if(d.error){$('out').innerHTML='<div class=card><p class=note>'+d.error+'. 먼저 [① 푸시 발송]에서 푸시를 보내세요.</p></div>';return}
  if(d.diff===undefined){$('out').innerHTML='<div class=card><p class=note>대조군이 없는 캠페인이라 비교할 수 없습니다. 대조군 분할을 켜고 다시 발송하세요.</p></div>';return}
  const sig=d.p_value<0.05;
  $('out').innerHTML=`<div class=card><h3>캠페인 #${d.campaign} 결과</h3><p class=note>푸시 받은 군 ${d.n_treat.toLocaleString()}명 / 대조군 ${d.n_ctrl.toLocaleString()}명 · 가정 효과 ${d.effect.toFixed(0)}% · 열람 ${d.opens}명</p>
  <div class=bar><span class=l>푸시 받은 군</span><div class=t><div class=f style="width:${f(d.rate_treat)}%"></div></div><span class=v>${f(d.rate_treat)}%</span></div>
  <div class=bar><span class=l>대조군</span><div class=t><div class="f c" style="width:${f(d.rate_ctrl)}%"></div></div><span class=v>${f(d.rate_ctrl)}%</span></div>
  <p>MAU 활성 비율 차이 <span class=big>${f(d.diff)}%p</span> <span class=note>(95% 신뢰구간 ${f(d.ci_low)} ~ ${f(d.ci_high)}%p, p=${d.p_value.toFixed(4)})</span></p>
  <p class="${sig?'sig':'nsig'}">${sig?'통계적으로 유의 (p<0.05)':'유의하지 않음 (p≥0.05)'}</p></div>
  <div class=card><h3>민감도: 가정 효과를 바꾸면?</h3><table><tr><th>가정 효과</th><th>활성 비율 차이</th><th>p값</th><th>이 표본에서</th></tr>${
   d.sens.map(s=>`<tr><td>${s.effect}%</td><td>${f(s.diff)}%p</td><td>${s.p_value.toFixed(4)}</td><td class="${s.p_value<0.05?'sig':'nsig'}">${s.p_value<0.05?'검출됨':'검출 안 됨'}</td></tr>`).join('')}</table>
  <p class=note>효과 0%는 "효과가 없을 때 검정이 유의하지 않아야 한다"는 점검입니다. 표본이 작으면 작은 효과는 검출되지 않을 수 있습니다.</p></div>`;
}
['camp','eff'].forEach(i=>$(i).addEventListener('input',load));load();
</script>"""

USER = HEAD + """<title>__BRAND__ (고객 화면 · 시연)</title><style>
*{box-sizing:border-box}
body{margin:0;background:#f4f6f9;font-family:'Noto Sans KR','Apple SD Gothic Neo','Malgun Gothic','Noto Sans CJK KR',sans-serif;color:#1b2433;line-height:1.5}
button{font:inherit;cursor:pointer}
.phone{background:#1c1c1f;border-radius:38px;padding:12px;max-width:380px;margin:8px auto}
.screen{background:#f2f4f8;border-radius:28px;height:660px;display:flex;flex-direction:column;overflow:hidden;position:relative;font-size:15px}
.screen.bigtext{font-size:18.5px}
.apphead{background:#fff;padding:.8em 1em;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #e3e7ee}
.apphead b{font-size:1.15em;color:#14264a}.apphead small{color:#5b6572;font-size:.75em}
.appbody{flex:1;overflow-y:auto;padding:.9em}
.tabbar{display:flex;background:#fff;border-top:1px solid #e3e7ee}
.tabbar button{flex:1;background:none;border:0;padding:.65em 0 .75em;color:#7a8494;font-size:.85em;position:relative}
.tabbar button.on{color:#14264a;font-weight:700}
.dot{position:absolute;top:.35em;right:28%;background:#d33;color:#fff;border-radius:99px;font-size:.7em;min-width:1.5em;padding:0 .3em;line-height:1.5em}
.pcard{background:#fff;border:1px solid #e1e6ee;border-radius:14px;padding:.85em .95em;margin-bottom:.7em}
.pcard h4{margin:0 0 .4em;font-size:1em;color:#14264a}.pcard.hl{border:2px solid #c25a00}
.pbtn{background:#14264a;color:#fff;border:0;border-radius:9px;padding:.5em .9em;font-size:.9em}
.pnote{font-size:.8em;color:#77808d}
.banner{position:absolute;left:.6em;right:.6em;top:.6em;background:#14264a;color:#fff;border-radius:14px;padding:.7em .85em;box-shadow:0 6px 16px #0005;animation:in .35s;z-index:5;cursor:pointer;font-size:.92em}
@keyframes in{from{transform:translateY(-30px);opacity:0}to{transform:none;opacity:1}}
.msg{background:#fff;border:1px solid #e1e6ee;border-radius:12px;padding:.7em .85em;margin-bottom:.55em;cursor:pointer;display:flex;gap:.6em}
.msg .u{width:.6em;height:.6em;border-radius:50%;background:#c25a00;margin-top:.45em;flex:none}.msg.read .u{background:#cfd5de}
.msg small{color:#888;font-size:.78em}
.sp{display:flex;align-items:center;gap:.5em;font-size:.9em;margin:.25em 0}.sp .n{width:3.2em}.sp .t{flex:1;height:.6em;background:#e7ebf1;border-radius:9px}.sp .f{height:100%;border-radius:9px;background:#5b78b4}.sp .a{width:5.2em;text-align:right;color:#445}
.sw{display:flex;justify-content:space-between;align-items:center;gap:.8em}
.toggle{width:3em;height:1.7em;border-radius:99px;border:0;background:#c3cad6;position:relative;flex:none}
.toggle::after{content:"";position:absolute;top:.2em;left:.2em;width:1.3em;height:1.3em;border-radius:50%;background:#fff;transition:.15s}
.toggle.on{background:#14264a}.toggle.on::after{left:1.5em}
</style>
<div class=phone><div class=screen id=screen>
 <div class=apphead><b>__BRAND__</b><small>__TAG__</small></div>
 <div id=banner></div><div class=appbody id=appbody></div>
 <div class=tabbar id=tabbar><button data-p=home class=on>홈</button><button data-p=alerts>알림<span id=dot></span></button><button data-p=benefit>혜택</button><button data-p=set>설정</button></div>
</div></div>
<script>
const CID=__CID__,PREVIEW=__PREVIEW__,AGE={'10-20s':'10~20대','30s':'30대','40s':'40대','50s+':'50대 이상'},CH={organic:'자연 유입',partner:'제휴 유입'};
const LANDS={event:{t:"첫 달 이벤트",d:"가입 첫 달 혜택을 확인해 보세요. (시연용 안내)"},report:{t:"소비 리포트",d:"이번 달 소비를 항목별로 한눈에 확인해 보세요. (합성 값)"},branch:{t:"영업점 · 앱 함께 쓰기",d:"가까운 영업점에서 앱 사용을 도와드립니다. 영업점과 앱을 함께 쓰면 더 편리합니다. (시연용 안내)"}};
const $=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let me=null,msgs=[],page="home",hl=null,bigText=false,seen=null;
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function spend(){const g=mulberry32(CID*31+7),v=[["식비",18+g()*30],["교통",4+g()*10],["쇼핑",6+g()*28],["기타",3+g()*14]].map(x=>[x[0],Math.round(x[1])]),mx=Math.max(...v.map(x=>x[1]));
 return v.map(x=>`<div class=sp><span class=n>${x[0]}</span><div class=t><div class=f style="width:${Math.round(x[1]/mx*100)}%"></div></div><span class=a>${x[1]}만 원</span></div>`).join('')}
function render(){
 $('screen').classList.toggle('bigtext',bigText);
 document.querySelectorAll('#tabbar button').forEach(b=>b.classList.toggle('on',b.dataset.p===page));
 const n=msgs.filter(m=>!m.opened_at).length;$('dot').innerHTML=n?`<span class=dot>${n}</span>`:'';let h='';
 if(page==="home")h=`<p style="margin:.2em 0 .7em"><b>안녕하세요, ${esc(me.name)}님</b><br><span class=pnote>${AGE[me.age_group]} · ${CH[me.channel]} (합성 고객)</span></p>
  <div class=pcard><h4>새 알림</h4>${n?`읽지 않은 알림이 <b>${n}건</b> 있습니다.`:'새 알림이 없습니다.'}<div style="margin-top:.5em"><button class=pbtn onclick="go('alerts')">알림 보기</button></div></div>
  <div class=pcard><h4>이번 달 소비 리포트 <span class=pnote>(합성 값)</span></h4>${spend()}</div>
  <p class=pnote>예금 · 적금 · 이체 등 금융 거래 기능은 이 시연에서 제외했습니다.</p>`;
 else if(page==="alerts")h='<p style="margin:.2em 0 .6em"><b>알림함</b></p>'+(msgs.length?msgs.map(x=>`<div class="msg ${x.opened_at?'read':''}" onclick="openMsg(${x.id})"><span class=u></span><div><b>${esc(x.title)}</b><br>${esc(x.body)}<br><small>${new Date(x.sent_at*1000).toLocaleTimeString('ko-KR')} · ${x.opened_at?'읽음':'읽지 않음'}</small></div></div>`).join('')
   :`<p class=pnote>${me.push_consent?'받은 알림이 없습니다.':'마케팅 알림 수신에 동의하지 않아 알림이 오지 않습니다. [설정]에서 바꿀 수 있습니다.'}</p>`);
 else if(page==="benefit")h='<p style="margin:.2em 0 .6em"><b>혜택 · 안내</b></p>'+Object.entries(LANDS).map(([k,v])=>`<div class="pcard ${hl===k?'hl':''}"><h4>${v.t}${hl===k?' <span class=pnote>· 알림에서 이동</span>':''}</h4>${v.d}${k==='report'?'<div style="margin-top:.5em">'+spend()+'</div>':''}</div>`).join('');
 else h=`<p style="margin:.2em 0 .6em"><b>설정</b></p>
  <div class=pcard><div class=sw><div><b>마케팅 알림 수신 동의</b><br><span class=pnote>끄면 혜택 · 이벤트 푸시가 발송 대상에서 제외됩니다.</span></div><button class="toggle ${me.push_consent?'on':''}" aria-label="마케팅 알림 수신 동의" onclick="toggleConsent()"></button></div></div>
  <div class=pcard><div class=sw><div><b>큰 글씨로 보기</b><br><span class=pnote>글씨를 크게 표시합니다.</span></div><button class="toggle ${bigText?'on':''}" aria-label="큰 글씨로 보기" onclick="bigText=!bigText;render()"></button></div></div>
  <div class=pcard><div class=sw><div><b>${PREVIEW?'관리자 미리보기 중':esc(me.name)+' 로그인 중'}</b><br><span class=pnote>${PREVIEW?'관리자 화면에서 로그아웃할 수 있습니다.':'로그아웃하면 로그인 화면으로 돌아갑니다.'}</span></div>${PREVIEW?'':'<button class=pbtn onclick="logout()">로그아웃</button>'}</div></div>
  <p class=pnote>이 화면은 시연용이며 실제 개인정보를 저장하지 않습니다.</p>`;
 $('appbody').innerHTML=h;
}
function go(p){page=p;if(p!=="benefit")hl=null;render()}
async function openMsg(id){const m=msgs.find(x=>x.id===id);if(!m)return;if(!m.opened_at){m.opened_at=Date.now()/1000;fetch('/api/open/'+id,{method:'POST'})}hl=m.land;page="benefit";$('banner').innerHTML='';render()}
async function toggleConsent(){const d=await (await fetch('/api/consent/'+CID,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({consent:!me.push_consent})})).json();me.push_consent=d.consent;render()}
function banner(m){const bn=$('banner');bn.innerHTML='';const d=document.createElement('div');d.className='banner';d.innerHTML=`<b>${esc(m.title)}</b><br>${esc(m.body)}`;d.onclick=()=>{bn.innerHTML='';go('alerts')};bn.append(d);setTimeout(()=>{if(bn.contains(d))bn.innerHTML=''},7000)}
async function poll(){  // 2초마다 서버에 새 알림이 있는지 묻는다
 try{const res=await fetch('/api/inbox/'+CID);
  if(res.status===401||res.status===403){location.href='/login';return}   // 세션이 끝났으면 로그인 화면으로
  const rows=await res.json();
  const top=rows.length?rows[0].id:0;
  if((seen!==null&&top>seen)||(seen===null&&rows.length&&!rows[0].opened_at&&Date.now()/1000-rows[0].sent_at<10))banner(rows[0]);  // 새 알림(또는 방금 온 알림)이면 배너 표시
  const changed=seen===null||top!==seen||rows.length!==msgs.length;seen=top;
  if(changed){msgs=rows;render()}
 }catch(e){}
 setTimeout(poll,2000);
}
async function logout(){await fetch('/api/logout',{method:'POST'});location.href='/login'}
document.querySelectorAll('#tabbar button').forEach(b=>b.onclick=()=>go(b.dataset.p));
(async()=>{me=await (await fetch('/api/customer/'+CID)).json();render();poll()})();
</script>"""


# 로그인 화면은 하나: 로그인과 회원가입. 관리자 화면으로 가는 버튼이나 링크는 없고, 계정의 역할로 넘어갈 화면이 정해진다.
LOGIN = HEAD + "<title>__BRAND__ 로그인 (시연)</title><style>" + CSS + """
.lnk{background:none;border:0;color:#14264a;text-decoration:underline;padding:0}</style><header><b>__BRAND__ (시연용 앱)</b></header>""" + WARN + """
<main><div class=card style='max-width:420px;margin:24px auto'>
<div id=fLogin><h3>로그인</h3>
 <label>아이디</label><input id=lid type=text autocomplete=off><div style='height:8px'></div>
 <label>비밀번호</label><input id=lpw type=password autocomplete=off>
 <div class=err id=lerr></div><button class=btn style='width:100%' onclick="login()">로그인</button>
 <p style='text-align:center;margin:12px 0 0'>계정이 없으신가요? <button class=lnk onclick="show('fSignup')">회원가입</button></p>
 <div class=hint style='margin-top:12px'><b>시연용 고객 계정</b> · 아이디에 고객 번호(예: 19, 55, 87), 비밀번호 <code>0000</code><br>실제로 쓰는 비밀번호는 입력하지 마세요. 5회 틀리면 30초 동안 잠깁니다.</div></div>
<div id=fSignup style='display:none'><h3>회원가입</h3>
 <label>아이디 (영문으로 시작, 4~20자)</label><input id=sid type=text autocomplete=off><div style='height:8px'></div>
 <label>비밀번호 (6자 이상)</label><input id=spw type=password autocomplete=off><div style='height:8px'></div>
 <label>비밀번호 확인</label><input id=spw2 type=password autocomplete=off><div style='height:8px'></div>
 <label>연령대</label><select id=sage><option value=''>선택 안 함</option><option value='10-20s'>10~20대</option><option value='30s'>30대</option><option value='40s'>40대</option><option value='50s+'>50대 이상</option></select><div style='height:10px'></div>
 <label style='display:flex;gap:8px;align-items:center;color:#1b2433'><input type=checkbox id=sconsent checked style='width:auto'> 혜택 · 이벤트 알림 수신 동의 (선택)</label>
 <div class=err id=serr></div><button class=btn style='width:100%' onclick="signup()">가입하기</button>
 <p style='text-align:center;margin:12px 0 0'><button class=lnk onclick="show('fLogin')">로그인으로 돌아가기</button></p>
 <div class=hint style='margin-top:12px'>시연용입니다. 실제 개인정보나 실제로 쓰는 비밀번호를 넣지 마세요. 가입하면 시연용 합성 고객 한 명이 계정에 연결됩니다.</div></div>
</div></main>
<script>
const $=id=>document.getElementById(id);
function show(f){$('fLogin').style.display=f==='fLogin'?'block':'none';$('fSignup').style.display=f==='fSignup'?'block':'none'}
async function post(url,body,errEl){
  const d=await (await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  if(d.error){$(errEl).textContent=d.error;return}
  location.href=d.next;   // 관리자 계정이면 /admin, 고객 계정이면 /app
}
function login(){post('/api/login',{id:$('lid').value,pw:$('lpw').value},'lerr');$('lpw').value=''}
function signup(){
  if($('spw').value!==$('spw2').value){$('serr').textContent='비밀번호 확인이 일치하지 않습니다.';return}
  post('/api/signup',{id:$('sid').value,pw:$('spw').value,age:$('sage').value,consent:$('sconsent').checked},'serr');
}
$('lpw').addEventListener('keydown',e=>{if(e.key==='Enter')login()});
</script>"""


@app.get("/")
def page_root():
    """접속하면 역할에 맞는 화면으로 보낸다. 로그인 전이거나 고객이면 고객 쪽으로만 간다."""
    role = session.get("role")
    return redirect("/admin" if role == "admin" else "/app" if role == "customer" else "/login")


@app.get("/login")
def page_login():
    return LOGIN.replace("__BRAND__", BRAND)


@app.get("/admin/login")       # 이전 주소 호환: 로그인 화면은 하나다
def page_admin_login():
    return redirect("/login")


@app.get("/admin")
@admin_required
def page_admin():
    return ADMIN


@app.get("/admin/results")
@admin_required
def page_results():
    return RESULTS


@app.get("/results")           # 이전 주소 호환
def page_results_old():
    return redirect("/admin/results")


def user_page(cid, preview):
    return (USER.replace("__CID__", str(cid)).replace("__BRAND__", BRAND)
            .replace("__PREVIEW__", "true" if preview else "false")
            .replace("__TAG__", "관리자 미리보기" if preview else "iM뱅크 적용 가정 · 시연용"))


@app.get("/app")
def page_my_app():
    """로그인한 고객 본인의 화면."""
    if session.get("role") != "customer":
        return redirect("/login")
    return user_page(session["cid"], preview=False)


@app.get("/app/<int:cid>")
def page_user(cid):
    """관리자 미리보기(또는 본인) 화면. 다른 고객의 화면은 볼 수 없다."""
    if not can_view(cid):
        return redirect("/login")
    con = db()
    ok = con.execute("SELECT 1 FROM customers WHERE customer_id=?", (cid,)).fetchone()
    con.close()
    if not ok:
        abort(404)
    return user_page(cid, preview=session.get("role") == "admin")


@app.get("/phone/<int:cid>")   # 이전 주소 호환
def page_phone(cid):
    return redirect(f"/app/{cid}")


init()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
