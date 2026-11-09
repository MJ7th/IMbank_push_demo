"""푸시 발송 웹앱 (운영자 콘솔 + 고객 휴대폰 화면)

실행:  pip install flask numpy pandas scipy
       python data/generate_data.py --sqlite
       python model/train.py
       python app/app.py          ->  http://localhost:5000

- /          운영자 콘솔: 위험 고객 조회 -> 문구 작성 -> 푸시 발송 (50:50 대조군 분할)
- /phone/<id> 고객 휴대폰 화면: 발송된 푸시가 2초 안에 알림으로 도착
- /results   A/B 결과: 가정한 효과를 넣은 시뮬레이션 + 비율 차이 검정

같은 Wi-Fi의 실제 스마트폰에서 http://<PC의 IP>:5000/phone/<고객ID> 를 열면 폰으로 푸시를 받아볼 수 있다.
※ 실제 은행 푸시(FCM 등)가 아니라 웹 기반 시연이다. 고객·문구·효과는 모두 합성/가정이다.
"""
import math
import os
import sqlite3
import time

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from scipy.stats import norm

DB = os.path.join(os.path.dirname(__file__), "..", "data", "imbank.sqlite")
app = Flask(__name__)

TEMPLATES = {
    "50s+": "고객님, 가까운 영업점과 앱을 함께 쓰시면 더 편리합니다. 앱 사용이 어려우시면 영업점에서 도와드립니다. (시연용 문구)",
    "10-20s": "가입 첫 달! 오늘 소비 리포트를 확인하고 용돈을 한눈에 관리해 보세요. (시연용 문구)",
    "default": "iM뱅크에서 필요한 금융 서비스를 확인해 보세요. (시연용 문구)",
}


def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def init():
    con = db()
    con.execute("""CREATE TABLE IF NOT EXISTS push_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT, campaign_id INTEGER, customer_id INTEGER,
        grp TEXT, title TEXT, body TEXT, sent_at REAL)""")
    con.commit()
    con.close()


def target_rows(con, age, channel, top):
    """필터를 적용한 뒤 위험 점수 상위 top% 고객 (필터 안에서의 상위)."""
    q = ("SELECT c.customer_id, c.age_group, c.channel, r.risk FROM customers c "
         "JOIN risk_scores r USING(customer_id) WHERE 1=1")
    args = []
    if age:
        q += " AND c.age_group=?"; args.append(age)
    if channel:
        q += " AND c.channel=?"; args.append(channel)
    rows = [dict(r) for r in con.execute(q + " ORDER BY r.risk DESC", args)]
    k = int(len(rows) * top / 100)
    return rows[:k], len(rows)


# ------------------------------------------------------------------ API
@app.get("/api/targets")
def api_targets():
    age, channel = request.args.get("age", ""), request.args.get("channel", "")
    top = float(request.args.get("top", 30))
    con = db()
    rows, pool = target_rows(con, age, channel, top)
    con.close()
    return jsonify(total=len(rows), pool=pool, sample=rows[:15],
                   template=TEMPLATES.get(age, TEMPLATES["default"]))


@app.post("/api/send")
def api_send():
    d = request.get_json()
    age, channel, top = d.get("age", ""), d.get("channel", ""), float(d.get("top", 30))
    title, body = d.get("title", "iM뱅크").strip(), d.get("body", "").strip()
    if not body:
        return jsonify(error="메시지 내용을 입력하세요"), 400
    con = db()
    rows, _ = target_rows(con, age, channel, top)
    if not rows:
        con.close()
        return jsonify(error="대상 고객이 없습니다"), 400
    campaign = (con.execute("SELECT COALESCE(MAX(campaign_id),0)+1 FROM push_log").fetchone()[0])
    rng = np.random.default_rng(campaign)  # 캠페인 번호를 시드로 -> 재현 가능한 무작위 분할
    use_control = d.get("control", True)
    groups = rng.random(len(rows)) < 0.5 if use_control else np.ones(len(rows), bool)
    now = time.time()
    data = [(campaign, r["customer_id"], "treatment" if g else "control", title, body, now)
            for r, g in zip(rows, groups)]
    con.executemany("INSERT INTO push_log(campaign_id,customer_id,grp,title,body,sent_at) VALUES(?,?,?,?,?,?)", data)
    con.commit()
    treated = [x[1] for x in data if x[2] == "treatment"]
    con.close()
    return jsonify(campaign=campaign, sent=len(treated), control=len(data) - len(treated),
                   examples=treated[:5])


@app.post("/api/reset")
def api_reset():
    con = db()
    con.execute("DELETE FROM push_log")
    con.commit()
    con.close()
    return jsonify(ok=True)


@app.get("/api/inbox/<int:cid>")
def api_inbox(cid):
    after = int(request.args.get("after", 0))
    con = db()
    rows = [dict(r) for r in con.execute(
        "SELECT id,title,body,sent_at FROM push_log WHERE customer_id=? AND grp='treatment' AND id>? ORDER BY id",
        (cid, after))]
    con.close()
    return jsonify(rows)


@app.get("/api/results")
def api_results():
    effect = float(request.args.get("effect", 8)) / 100  # 가정: 푸시를 받은 이탈 위험 고객 중 활성화로 돌아서는 비율
    con = db()
    cid = request.args.get("campaign")
    if not cid:
        r = con.execute("SELECT MAX(campaign_id) FROM push_log").fetchone()[0]
        if r is None:
            con.close()
            return jsonify(error="아직 발송한 캠페인이 없습니다")
        cid = r
    rows = con.execute(
        "SELECT p.customer_id, p.grp, c.churn FROM push_log p JOIN customers c USING(customer_id) "
        "WHERE p.campaign_id=?", (cid,)).fetchall()
    con.close()
    rng = np.random.default_rng(int(cid) * 7919)
    stats = {"treatment": [0, 0], "control": [0, 0]}  # [활성 수, 전체]
    for r in rows:
        active = 1 - r["churn"]
        if r["grp"] == "treatment" and r["churn"] == 1 and rng.random() < effect:
            active = 1  # 가정한 효과 적용
        stats[r["grp"]][0] += active
        stats[r["grp"]][1] += 1
    (a1, n1), (a0, n0) = stats["treatment"], stats["control"]
    out = dict(campaign=int(cid), effect=effect * 100, n_treat=n1, n_ctrl=n0)
    if n1 and n0:
        p1, p0 = a1 / n1, a0 / n0
        pp = (a1 + a0) / (n1 + n0)
        se0 = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
        se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        z = (p1 - p0) / se0 if se0 else 0
        out.update(rate_treat=p1 * 100, rate_ctrl=p0 * 100, diff=(p1 - p0) * 100,
                   ci_low=(p1 - p0 - 1.96 * se) * 100, ci_high=(p1 - p0 + 1.96 * se) * 100,
                   p_value=float(2 * (1 - norm.cdf(abs(z)))))
    return jsonify(out)


# ------------------------------------------------------------------ pages
CSS = """
body{font-family:'Noto Sans KR','Noto Sans CJK KR','Malgun Gothic',sans-serif;margin:0;background:#f4f6f9;color:#1b2433}
header{background:#14264a;color:#fff;padding:14px 24px;display:flex;gap:24px;align-items:center}
header b{font-size:18px}header a{color:#cfd9ec;text-decoration:none}header a:hover{color:#fff}
main{max-width:980px;margin:20px auto;padding:0 16px}
.card{background:#fff;border:1px solid #d9dee6;border-radius:12px;padding:18px 20px;margin-bottom:16px}
label{font-size:13px;color:#555;display:block;margin-bottom:4px}
select,input,textarea{font:inherit;padding:7px 9px;border:1px solid #b9c2d0;border-radius:8px;width:100%;box-sizing:border-box}
.row{display:flex;gap:12px}.row>div{flex:1}
button{background:#c25a00;color:#fff;border:0;padding:10px 18px;border-radius:8px;font:inherit;font-weight:700;cursor:pointer}
button:disabled{background:#aaa}
table{border-collapse:collapse;width:100%;font-size:14px}th{background:#dce6f2}th,td{border:1px solid #c4ccd8;padding:5px 8px;text-align:left}
.note{font-size:13px;color:#666}.ok{background:#e9f7ee;border:1px solid #9bd3ac;padding:10px;border-radius:8px}
.big{font-size:28px;font-weight:700;color:#14264a}
"""
NAV = "<header><b>iM뱅크 푸시 콘솔 (시연)</b><a href='/'>발송</a><a href='/results'>A/B 결과</a></header>"

ADMIN = """<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>푸시 발송 콘솔</title><style>""" + CSS + """</style>""" + NAV + """
<main>
<div class=card><h3 style='margin-top:0'>1. 대상 고객 선택</h3>
<div class=row>
<div><label>연령대</label><select id=age><option value=''>전체</option><option value='10-20s'>10~20대</option><option value='30s'>30대</option><option value='40s'>40대</option><option value='50s+'>50대 이상</option></select></div>
<div><label>유입 경로</label><select id=channel><option value=''>전체</option><option value='organic'>자연 유입</option><option value='partner'>제휴 유입</option></select></div>
<div><label>이탈 위험 상위 (%)</label><input id=top type=number value=30 min=1 max=100></div>
</div>
<p id=count class=note></p>
<table id=tbl><thead><tr><th>고객 ID</th><th>연령대</th><th>유입</th><th>이탈 위험 점수</th></tr></thead><tbody></tbody></table>
<p class=note>※ 이탈 위험 점수는 가입 후 첫 7일 행동으로 예측한 확률이다. 위 고객은 모두 합성 데이터다.</p></div>

<div class=card><h3 style='margin-top:0'>2. 푸시 문구 작성</h3>
<label>제목</label><input id=title value='iM뱅크'><br><br>
<label>내용 (연령대를 고르면 시연용 문구가 채워집니다)</label><textarea id=body rows=3></textarea><br><br>
<label><input type=checkbox id=ctrl checked style='width:auto'> 대조군 분할 (대상의 절반에게는 보내지 않고 효과 비교용으로 남김)</label></div>

<div class=card><h3 style='margin-top:0'>3. 발송</h3>
<button id=send>푸시 발송</button> <button id=reset style='background:#666'>발송 기록 초기화</button>
<div id=result style='margin-top:12px'></div></div>
</main>
<script>
const $=id=>document.getElementById(id);
async function load(){
  const q=new URLSearchParams({age:$('age').value,channel:$('channel').value,top:$('top').value});
  const d=await (await fetch('/api/targets?'+q)).json();
  $('count').textContent=`대상 ${d.total.toLocaleString()}명 (조건에 맞는 고객 ${d.pool.toLocaleString()}명 중)`;
  $('tbl').tBodies[0].innerHTML=d.sample.map(r=>`<tr><td>${r.customer_id}</td><td>${r.age_group}</td><td>${r.channel}</td><td>${r.risk.toFixed(3)}</td></tr>`).join('');
  $('body').value=d.template;
}
['age','channel','top'].forEach(i=>$(i).addEventListener('change',load));
$('send').onclick=async()=>{
  $('send').disabled=true;
  const r=await fetch('/api/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
    age:$('age').value,channel:$('channel').value,top:$('top').value,title:$('title').value,body:$('body').value,control:$('ctrl').checked})});
  const d=await r.json();
  $('send').disabled=false;
  if(d.error){$('result').innerHTML=`<div class=note style='color:#b00'>${d.error}</div>`;return}
  const links=d.examples.map(c=>`<a href='/phone/${c}' target=_blank>고객 ${c}</a>`).join(' · ');
  $('result').innerHTML=`<div class=ok>캠페인 #${d.campaign}: 푸시 ${d.sent.toLocaleString()}명 발송, 대조군 ${d.control.toLocaleString()}명<br>받은 화면 보기: ${links}</div>`;
};
$('reset').onclick=async()=>{if(confirm('모든 발송 기록을 지울까요?')){await fetch('/api/reset',{method:'POST'});$('result').innerHTML='<div class=note>초기화했습니다.</div>'}};
load();
</script>"""

PHONE = """<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>고객 휴대폰 화면</title><style>""" + CSS + """
body{background:#222}.phone{max-width:380px;margin:20px auto;background:#f4f6f9;border-radius:28px;min-height:640px;padding:18px;box-shadow:0 8px 30px #0008}
.banner{position:sticky;top:8px;background:#14264a;color:#fff;border-radius:14px;padding:12px 14px;margin-bottom:12px;box-shadow:0 4px 12px #0004;animation:in .3s}
@keyframes in{from{transform:translateY(-20px);opacity:0}to{transform:none;opacity:1}}
.msg{background:#fff;border-radius:12px;padding:12px;margin-bottom:10px;border:1px solid #dde3ec}
.msg small{color:#777}</style>
<div class=phone><div style='display:flex;justify-content:space-between'><b>iM뱅크</b><small>고객 {{cid}} (시연)</small></div>
<p class=note>운영자 콘솔에서 푸시를 보내면 이 화면에 알림이 도착합니다.</p>
<button id=perm style='background:#555;padding:6px 10px;font-size:12px'>브라우저 알림 허용</button>
<div id=banner></div><h4>알림 목록</h4><div id=list><p class=note id=empty>받은 알림이 없습니다.</p></div></div>
<script>
let last=0;const cid={{cid}};
document.getElementById('perm').onclick=()=>{if('Notification' in window)Notification.requestPermission()};
async function poll(){
  try{
    const rows=await (await fetch('/api/inbox/'+cid+'?after='+last)).json();
    for(const m of rows){
      last=Math.max(last,m.id);
      document.getElementById('empty')?.remove();
      const t=new Date(m.sent_at*1000).toLocaleTimeString('ko-KR');
      document.getElementById('list').insertAdjacentHTML('afterbegin',`<div class=msg><b>${m.title}</b><br>${m.body}<br><small>${t}</small></div>`);
      const bn=document.getElementById('banner');
      bn.innerHTML=`<div class=banner><b>${m.title}</b><br>${m.body}</div>`;setTimeout(()=>bn.innerHTML='',6000);
      if('Notification' in window&&Notification.permission==='granted')new Notification(m.title,{body:m.body});
    }
  }catch(e){}
  setTimeout(poll,2000);
}
poll();
</script>"""

RESULTS = """<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>A/B 결과</title><style>""" + CSS + """</style>""" + NAV + """
<main><div class=card><h3 style='margin-top:0'>A/B 실험 결과 (가정 효과 적용 시뮬레이션)</h3>
<div class=row><div><label>캠페인 번호 (비우면 최근)</label><input id=camp></div>
<div><label>가정한 효과: 푸시를 받은 이탈 고객 중 활성화로 돌아서는 비율 (%)</label><input id=eff type=number value=8 min=0 max=100></div></div>
<p class=note>※ 효과는 실제 측정값이 아니라 위에서 입력한 <b>가정값</b>이다. 값을 바꿔 가며 어느 정도 효과부터 통계적으로 검출되는지 본다.</p></div>
<div id=out class=card></div></main>
<script>
const $=id=>document.getElementById(id);
async function load(){
  const d=await (await fetch('/api/results?campaign='+$('camp').value+'&effect='+$('eff').value)).json();
  if(d.error){$('out').innerHTML='<p class=note>'+d.error+' 먼저 [발송]에서 푸시를 보내세요.</p>';return}
  const sig=d.p_value<0.05?'통계적으로 유의 (p<0.05)':'유의하지 않음 (p≥0.05)';
  $('out').innerHTML=`<p class=note>캠페인 #${d.campaign} · 처치군 ${d.n_treat.toLocaleString()}명 / 대조군 ${d.n_ctrl.toLocaleString()}명</p>
  <table><tr><th>구분</th><th>30일 활성화율</th></tr><tr><td>푸시 받은 군</td><td>${d.rate_treat?.toFixed(1)}%</td></tr><tr><td>대조군</td><td>${d.rate_ctrl?.toFixed(1)}%</td></tr></table>
  <p>차이 <span class=big>${d.diff?.toFixed(1)}%p</span> <span class=note>(95% 신뢰구간 ${d.ci_low?.toFixed(1)} ~ ${d.ci_high?.toFixed(1)}%p, p=${d.p_value?.toFixed(4)})</span></p><p><b>${sig}</b></p>`;
}
['camp','eff'].forEach(i=>$(i).addEventListener('change',load));load();
</script>"""


@app.get("/")
def admin():
    return ADMIN


@app.get("/phone/<int:cid>")
def phone(cid):
    return render_template_string(PHONE, cid=cid)


@app.get("/results")
def results():
    return RESULTS


init()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
