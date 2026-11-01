"""
follower_web.py - password-protected, phone-friendly pages for the follower system.

  /followers   add / edit / delete follower rows, global demo|real switch, live status
  /verdict     honest statistics of the pre-registered zone test (negatives are shown)
  /capacity    follower-loop lag, mirror delay, connected followers, DB queue depth

Auth: ADMIN_PASSWORD (constant-time compare) + signed session cookie (secret derived from
ENCRYPTION_KEY) + login throttling. If ADMIN_PASSWORD or ENCRYPTION_KEY is unset the pages
refuse to load. Tokens are Fernet-encrypted before they reach the DB and are never sent
back to the browser (only the last 4 characters are shown).
Suggested token scopes: trade + read only (no payments, no admin).
"""
import hashlib
import hmac
import logging
import os
import re
import secrets
import time
from typing import Dict, List, Optional

from flask import Blueprint, make_response, redirect, render_template_string, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

import db
import follower_config as fc
import token_crypto
import verdict as vd

logger = logging.getLogger("follower.web")
bp = Blueprint("follower_web", __name__)

COOKIE = "fsess"
SESSION_SECONDS = 12 * 3600
_attempts: Dict[str, List[float]] = {}        # ip -> [failures, locked_until]
_global_fail: List[float] = []                # timestamps of recent failures (all IPs)

MESSAGES = {
    "saved": "Saved. Applies at the next connection (no restart needed).",
    "added": "Follower added. It connects at the next zone session.",
    "deleted": "Follower deleted.",
    "mode": "Mode switched for ALL followers (applies at the next connection).",
    "badinput": "Check the fields: label 1-40 chars, token 8-200 chars, app id letters/digits.",
    "badconfirm": "Confirmation text did not match exactly - mode unchanged.",
    "full": f"Maximum {fc.MAX_FOLLOWERS} followers.",
    "needtoken": "A token is required for a new follower.",
    "dberr": "Database error - nothing was changed. Try again in a moment.",
}


# ── auth helpers ─────────────────────────────────────────────────────────────
def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(token_crypto.session_secret(), salt="follower-session")


def _client_ip() -> str:
    xff = request.headers.get("X-Forwarded-For", "")
    return (xff.split(",")[-1].strip() if xff else "") or (request.remote_addr or "?")


def _session_id() -> Optional[str]:
    raw = request.cookies.get(COOKIE)
    if not raw:
        return None
    try:
        data = _serializer().loads(raw, max_age=SESSION_SECONDS)
        return data.get("sid")
    except (BadSignature, SignatureExpired):
        return None


def _csrf(sid: str) -> str:
    return hmac.new(token_crypto.session_secret(), b"csrf|" + sid.encode(), hashlib.sha256).hexdigest()[:32]


def _ready() -> bool:
    return fc.web_enabled() and token_crypto.available()


def _locked(ip: str) -> bool:
    now = time.time()
    _global_fail[:] = [t for t in _global_fail if now - t < 600]
    if len(_global_fail) >= 30:
        return True
    a = _attempts.get(ip)
    return bool(a and a[1] > now)


def _note_fail(ip: str) -> None:
    now = time.time()
    _global_fail.append(now)
    a = _attempts.setdefault(ip, [0, 0.0])
    a[0] += 1
    if a[0] >= 5:
        a[0] = 0
        a[1] = now + 600
    if len(_attempts) > 500:
        _attempts.clear()


def _password_ok(candidate: str) -> bool:
    a = hashlib.sha256(candidate.encode()).digest()
    b = hashlib.sha256(fc.ADMIN_PASSWORD.encode()).digest()
    return hmac.compare_digest(a, b)


def _respond(html: str, status: int = 200):
    r = make_response(html, status)
    r.headers["Cache-Control"] = "no-store"
    r.headers["X-Frame-Options"] = "DENY"
    r.headers["Referrer-Policy"] = "no-referrer"
    return r


def _guard():
    """Returns (response, sid). response is not None when the request must stop here."""
    if not _ready():
        return _respond(_page("Not configured",
                              "<p>Set <b>ADMIN_PASSWORD</b> and <b>ENCRYPTION_KEY</b> on Render, then redeploy.</p>",
                              nav=False), 503), None
    sid = _session_id()
    if not sid:
        return redirect("/follower-login"), None
    return None, sid


def _csrf_ok(sid: str) -> bool:
    return hmac.compare_digest(request.form.get("csrf", ""), _csrf(sid))


# ── templates ────────────────────────────────────────────────────────────────
BASE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex">
{% if refresh %}<meta http-equiv="refresh" content="{{ refresh }}">{% endif %}
<title>{{ title }}</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--mut:#667085;--line:#e3e6ea;--acc:#2457d6;--bad:#b42318;--good:#067647;--warn:#b54708}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#181b21;--fg:#e8eaed;--mut:#9aa3af;--line:#2a2f38;--acc:#6b93ff;--bad:#ff7b72;--good:#4ade80;--warn:#fbbf24}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.45 system-ui,sans-serif}
.wrap{max-width:640px;margin:0 auto;padding:12px}
nav{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}nav a,nav button{padding:9px 12px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--fg);text-decoration:none;font-size:15px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin-bottom:12px;overflow-x:auto}
h1{font-size:20px;margin:4px 0 12px}h2{font-size:17px;margin:0 0 8px}
label{display:block;font-size:13px;color:var(--mut);margin:10px 0 3px}
input[type=text],input[type=password]{width:100%;padding:11px;border:1px solid var(--line);border-radius:10px;background:var(--bg);color:var(--fg);font-size:16px}
.row{display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap}
button.p{padding:11px 16px;border:0;border-radius:10px;background:var(--acc);color:#fff;font-size:16px}
button.d{padding:11px 16px;border:1px solid var(--bad);border-radius:10px;background:transparent;color:var(--bad);font-size:16px}
.mut{color:var(--mut);font-size:13px}.msg{padding:10px;border-radius:10px;background:var(--card);border:1px solid var(--acc);margin-bottom:12px}
.bad{color:var(--bad)}.good{color:var(--good)}.warn{color:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:14px;margin-top:8px}.kv b{font-weight:500;color:var(--mut)}
</style></head><body><div class="wrap">
{% if nav %}<nav><a href="/followers">Followers</a><a href="/verdict">Verdict</a><a href="/capacity">Capacity</a>
<form method="post" action="/follower-logout" style="margin:0"><button type="submit">Log out</button></form></nav>{% endif %}
{% if msg %}<div class="msg">{{ msg }}</div>{% endif %}
{{ body|safe }}</div></body></html>"""


def _page(title: str, body: str, nav: bool = True, msg: str = "", refresh: int = 0) -> str:
    return render_template_string(BASE, title=title, body=body, nav=nav, msg=msg, refresh=refresh)


def _esc(x) -> str:
    return render_template_string("{{ x }}", x="" if x is None else x)


def _n(x, d=2, pre="") -> str:
    return "-" if x is None else f"{pre}{x:.{d}f}"


def _pct(x, d=1) -> str:
    return "-" if x is None else f"{x * 100:.{d}f}%"


# ── login / logout ───────────────────────────────────────────────────────────
@bp.route("/follower-login", methods=["GET", "POST"])
def login():
    if not _ready():
        return _guard()[0]
    err = ""
    if request.method == "POST":
        ip = _client_ip()
        if _locked(ip):
            return _respond(_page("Login", "<div class='card'>Too many attempts. Try again in a few minutes.</div>",
                                  nav=False), 429)
        if _password_ok(request.form.get("password", "")):
            _attempts.pop(ip, None)
            resp = redirect("/followers")
            sid = secrets.token_hex(16)
            secure = request.is_secure or request.headers.get("X-Forwarded-Proto", "") == "https"
            resp.set_cookie(COOKIE, _serializer().dumps({"sid": sid}), max_age=SESSION_SECONDS,
                            httponly=True, samesite="Strict", secure=secure)
            return resp
        _note_fail(ip)
        logger.warning("FOLLOWER web: failed login")
        time.sleep(0.8)
        err = "<p class='bad'>Wrong password.</p>"
    body = ("<h1>Follower admin</h1><div class='card'><form method='post'>"
            "<label>Password</label><input type='password' name='password' autocomplete='current-password' autofocus>"
            f"<div class='row'><button class='p' type='submit'>Log in</button></div>{err}</form></div>")
    return _respond(_page("Login", body, nav=False))


@bp.route("/follower-logout", methods=["POST"])
def logout():
    resp = redirect("/follower-login")
    resp.delete_cookie(COOKIE)
    return resp


# ── followers ────────────────────────────────────────────────────────────────
def _follower_card(f: dict, csrf: str) -> str:
    last4 = f.get("token_last4") or ""
    return render_template_string("""
<div class="card"><form method="post" action="/followers/save">
<input type="hidden" name="csrf" value="{{ csrf }}"><input type="hidden" name="id" value="{{ f.id }}">
<h2>{{ f.label }}{% if not f.enabled %} <span class="mut">(off)</span>{% endif %}</h2>
<label>Label</label><input type="text" name="label" value="{{ f.label }}" maxlength="40">
<label>API token (write-only){% if last4 %} - saved: ••••{{ last4 }}{% endif %}</label>
<input type="password" name="token" placeholder="leave blank to keep the saved token" autocomplete="off">
<label>App ID</label><input type="text" name="app_id" value="{{ f.app_id }}" maxlength="32">
<div class="row"><label style="margin:0"><input type="checkbox" name="enabled" value="1" {{ 'checked' if f.enabled }}> Enabled</label></div>
<div class="kv"><b>Balance</b><span>{{ bal }}</span><b>Flat stake</b><span>{{ stake }}</span>
<b>Last window</b><span>{{ last }}</span></div>
<div class="row"><button class="p" type="submit">Save</button></div></form>
<form method="post" action="/followers/delete" onsubmit="return confirm('Delete this follower?')">
<input type="hidden" name="csrf" value="{{ csrf }}"><input type="hidden" name="id" value="{{ f.id }}">
<div class="row"><button class="d" type="submit">Delete</button></div></form></div>""",
                                  f=f, csrf=csrf, last4=last4, bal=_n(f.get("last_balance")),
                                  stake=_n(fc.STAKE),
                                  last=f.get("last_window_result") or "-")


@bp.route("/followers", methods=["GET"])
def followers():
    stop, sid = _guard()
    if stop is not None:
        return stop
    csrf = _csrf(sid)
    msg = MESSAGES.get(request.args.get("m", ""), "")
    try:
        rows = db.list_followers()
        mode = db.get_mode()
        err = ""
    except Exception as exc:
        rows, mode, err = [], "demo", f"<div class='card bad'>Database not reachable ({_esc(type(exc).__name__)}).</div>"
    target = "real" if mode == "demo" else "demo"
    phrase = "SWITCH TO " + target.upper()
    mode_card = render_template_string("""
<div class="card"><h2>Global mode: <span class="{{ 'bad' if mode=='real' else 'good' }}">{{ mode|upper }}</span></h2>
<p class="mut">One switch for ALL followers. {% if mode=='demo' %}Real mode trades real money.{% endif %}</p>
<form method="post" action="/followers/mode"><input type="hidden" name="csrf" value="{{ csrf }}">
<input type="hidden" name="target" value="{{ target }}">
<label>Type <b>{{ phrase }}</b> to confirm</label><input type="text" name="confirm" autocomplete="off" autocapitalize="characters">
<div class="row"><button class="d" type="submit">Switch to {{ target|upper }}</button></div></form></div>""",
                                      mode=mode, target=target, phrase=phrase, csrf=csrf)
    cards = "".join(_follower_card(f, csrf) for f in rows)
    add = render_template_string("""
<div class="card"><h2>Add follower</h2><form method="post" action="/followers/save">
<input type="hidden" name="csrf" value="{{ csrf }}"><input type="hidden" name="id" value="">
<label>Label</label><input type="text" name="label" maxlength="40">
<label>API token (trade + read scopes only)</label><input type="password" name="token" autocomplete="off">
<label>App ID</label><input type="text" name="app_id" value="{{ app }}" maxlength="32">
<div class="row"><label style="margin:0"><input type="checkbox" name="enabled" value="1" checked> Enabled</label></div>
<div class="row"><button class="p" type="submit">Add</button></div></form></div>""",
                                 csrf=csrf, app=os.environ.get("DERIV_APP_ID", "1089"))
    note = ("<p class='mut'>Changes apply at the next connection (before the next zone session). "
            "Followers only connect around zone sessions. The zones are a hypothesis under test, not a proven edge.</p>")
    return _respond(_page("Followers", f"<h1>Followers</h1>{err}{mode_card}{cards}{add}{note}", msg=msg))


@bp.route("/followers/save", methods=["POST"])
def save_follower():
    stop, sid = _guard()
    if stop is not None:
        return stop
    if not _csrf_ok(sid):
        return redirect("/followers?m=badinput")
    label = request.form.get("label", "").strip()
    token = request.form.get("token", "").strip()
    app_id = request.form.get("app_id", "").strip() or os.environ.get("DERIV_APP_ID", "1089")
    enabled = request.form.get("enabled") == "1"
    fid = request.form.get("id", "").strip()
    if not (1 <= len(label) <= 40) or not re.fullmatch(r"[A-Za-z0-9_\-]{1,32}", app_id) \
            or (token and not re.fullmatch(r"[A-Za-z0-9_\-\.=+/]{8,200}", token)):
        return redirect("/followers?m=badinput")
    try:
        if fid:
            db.update_follower(int(fid), label, app_id, enabled,
                               token_enc=token_crypto.encrypt(token) if token else None,
                               last4=token[-4:] if token else None)
            return redirect("/followers?m=saved")
        if not token:
            return redirect("/followers?m=needtoken")
        if db.count_followers() >= fc.MAX_FOLLOWERS:
            return redirect("/followers?m=full")
        db.add_follower(label, token_crypto.encrypt(token), token[-4:], app_id, enabled)
        return redirect("/followers?m=added")
    except Exception as exc:
        logger.warning("FOLLOWER web: save failed (%s)", type(exc).__name__)
        return redirect("/followers?m=dberr")


@bp.route("/followers/delete", methods=["POST"])
def delete_follower():
    stop, sid = _guard()
    if stop is not None:
        return stop
    if not _csrf_ok(sid):
        return redirect("/followers?m=badinput")
    try:
        db.delete_follower(int(request.form.get("id", "0")))
        return redirect("/followers?m=deleted")
    except Exception as exc:
        logger.warning("FOLLOWER web: delete failed (%s)", type(exc).__name__)
        return redirect("/followers?m=dberr")


@bp.route("/followers/mode", methods=["POST"])
def switch_mode():
    stop, sid = _guard()
    if stop is not None:
        return stop
    if not _csrf_ok(sid):
        return redirect("/followers?m=badinput")
    target = request.form.get("target", "")
    if target not in ("demo", "real") or request.form.get("confirm", "").strip() != "SWITCH TO " + target.upper():
        return redirect("/followers?m=badconfirm")
    try:
        db.set_setting("mode", target)
        logger.warning("FOLLOWER mode switched to %s (global, applies at next connection)", target.upper())
        return redirect("/followers?m=mode")
    except Exception as exc:
        logger.warning("FOLLOWER web: mode switch failed (%s)", type(exc).__name__)
        return redirect("/followers?m=dberr")


# ── verdict ──────────────────────────────────────────────────────────────────
def _vrow(name: str, s: dict) -> str:
    cls = {vd.PROFIT: "good", vd.LOSS: "bad"}.get(s["state"], "warn")
    ci = "-" if s.get("ci_low") is None else f"{s['ci_low']:+.3f} .. {s['ci_high']:+.3f}"
    return (f"<tr><td>{_esc(name)}</td><td>{s['n']}</td><td>{_pct(s.get('win_rate'))}</td>"
            f"<td>{_n(s.get('avg_ratio'), 2)}x</td><td>{_pct(s.get('break_even'))}</td>"
            f"<td>{_n(s.get('mean'), 3)}</td><td>{ci}</td><td class='{cls}'><b>{_esc(s['state'])}</b></td></tr>")


@bp.route("/verdict", methods=["GET"])
def verdict_page():
    stop, _sid = _guard()
    if stop is not None:
        return stop
    try:
        rows = db.verdict_rows()
        minute_rows = db.load_minute_rows()
        snap = db.latest_snapshot()
        pnl = db.follower_pnl()
        labels = {f["id"]: f["label"] for f in db.list_followers()}
    except Exception as exc:
        return _respond(_page("Verdict", f"<h1>Verdict</h1><div class='card bad'>Database not reachable "
                                          f"({_esc(type(exc).__name__)}).</div>"))
    res = vd.by_source(rows, fc.VERDICT_MIN_TRADES)
    zone_minutes = set(range(fc.DEFAULT_ZONE_START, fc.DEFAULT_ZONE_END + 1))
    if snap:
        zone_minutes |= {m for m, t in enumerate((snap["zones"] or {}).get("tags") or []) if t}
    oz = vd.out_of_zone(minute_rows, zone_minutes)
    cmp_ = vd.compare(res["all"], oz)
    tot_n = sum(int(r[0]) for r in minute_rows)
    tot_w = sum(int(r[1]) for r in minute_rows)
    tot_r = sum(float(r[3]) for r in minute_rows)
    tot_u = sum(float(r[4]) for r in minute_rows)
    hist = ("no main-account data yet" if not tot_n else
            f"{tot_n} main-account trades: win rate {_pct(tot_w / tot_n)} vs break-even "
            f"{_pct(1 / (tot_r / tot_n)) if tot_r else '-'}; average profit per stake {tot_u / tot_n:+.3f}")
    head = (f"<div class='card'><p><b>Pre-registered hypothesis (not a finding):</b> minutes "
            f"{fc.DEFAULT_ZONE_START}-{fc.DEFAULT_ZONE_END} of every hour. Digit outcomes are random-number "
            f"generated, so no time zone may exist. This page reports whatever the data shows, including "
            f"negative results.</p><p class='mut'>Collector history: {hist}.</p>"
            f"<p class='mut'>Counted: follower trades placed after a zone snapshot only, one observation per "
            f"main-account signal (not per follower). Verdict needs {fc.VERDICT_MIN_TRADES}+ signals. "
            f"Profit is in stakes per trade (win = payout-1, loss = -1) with a 95% CI.</p></div>")
    table = ("<div class='card'><h2>Followed signals</h2><table><tr><th>Group</th><th>Signals</th><th>Win</th>"
             "<th>Avg payout</th><th>Break-even</th><th>Profit/stake</th><th>95% CI</th><th>Verdict</th></tr>"
             + _vrow("All", res["all"]) + _vrow("Default zone", res["default"]) + _vrow("Engine zones", res["engine"])
             + "</table></div>")
    if oz.get("n"):
        oz_html = (f"<div class='card'><h2>In-zone vs out-of-zone</h2><table><tr><th>Group</th><th>Trades</th>"
                   f"<th>Win</th><th>Break-even</th><th>Profit/stake</th></tr>"
                   f"<tr><td>In-zone (followed signals)</td><td>{res['all']['n']}</td><td>{_pct(res['all'].get('win_rate'))}</td>"
                   f"<td>{_pct(res['all'].get('break_even'))}</td><td>{_n(res['all'].get('mean'), 3)}</td></tr>"
                   f"<tr><td>Out-of-zone (main account)</td><td>{oz['n']}</td><td>{_pct(oz.get('win_rate'))}</td>"
                   f"<td>{_pct(oz.get('break_even'))}</td><td>{_n(oz.get('mean'), 3)}</td></tr></table>")
        if cmp_:
            oz_html += (f"<p class='mut'>Difference in profit/stake (in minus out): {cmp_['diff']:+.3f}, "
                        f"95% CI {cmp_['ci_low']:+.3f} .. {cmp_['ci_high']:+.3f} (out-of-zone CI is approximate).</p>")
        oz_html += "</div>"
    else:
        oz_html = ""
    pn = "".join(f"<tr><td>{_esc(labels.get(p['follower_id'], '#' + str(p['follower_id'])))}</td><td>{p['trades']}</td>"
                 f"<td>{p['wins']}</td><td class='{'good' if p['pnl'] > 0 else 'bad'}'>{p['pnl']:+.2f}</td></tr>" for p in pnl)
    pnl_html = ("<div class='card'><h2>Follower P&amp;L (separate from the verdict)</h2><table><tr><th>Follower</th>"
                "<th>Trades</th><th>Wins</th><th>P&amp;L</th></tr>" + (pn or "<tr><td colspan=4>none yet</td></tr>")
                + "</table></div>")
    return _respond(_page("Verdict", "<h1>Verdict</h1>" + head + table + oz_html + pnl_html))


# ── capacity ─────────────────────────────────────────────────────────────────
@bp.route("/capacity", methods=["GET"])
def capacity():
    stop, _sid = _guard()
    if stop is not None:
        return stop
    try:
        import follower_manager as fm
        m = dict(fm.METRICS)
        plans = []
        mgr = fm._MGR
        if mgr is not None:
            for hk, plan in sorted(mgr.plans.items()):
                for s in plan.sessions:
                    plans.append(f"H{hk} min {s['start']}-{s['end']} {'valid' if s['valid'] else 'SKIPPED (snapshot too late)'}")
    except Exception:
        m, plans = {"running": False}, []
    q = db.queue_stats()
    st = db.status()
    kv = [("Follower thread", "running" if m.get("running") else "NOT running"),
          ("Loop lag avg / max", f"{m.get('lag_avg_ms', 0)} / {m.get('lag_max_ms', 0)} ms"),
          ("Mirror delay avg / max", "-" if m.get("delay_avg_ms") is None else f"{m['delay_avg_ms']} / {m['delay_max_ms']} ms"),
          ("Connected followers", m.get("connected", 0)),
          ("Signals seen / old-dropped / not in zone", f"{m.get('signals', 0)} / {m.get('dropped_old', 0)} / {m.get('not_in_zone', 0)}"),
          ("Buys sent / errors", f"{m.get('buys_sent', 0)} / {m.get('buy_errors', 0)}"),
          ("Deriv server-time offset", f"{m.get('offset_s', 0)} s"),
          ("DB reachable", "yes" if q["db_ok"] else "no / unknown"),
          ("DB queue (pending / spilled to file)", f"{q['pending']} / {q['spilled_lines']}")]
    if st.get("last_error") and not q["db_ok"]:
        kv.append(("Last DB error", st["last_error"]))
    body = render_template_string("""<h1>Capacity</h1><div class="card"><div class="kv">
{% for k,v in kv %}<b>{{ k }}</b><span>{{ v }}</span>{% endfor %}</div></div>
<div class="card"><h2>Sessions (this hour and next)</h2>{% for p in plans %}<div>{{ p }}</div>{% else %}<div class="mut">none yet</div>{% endfor %}</div>
<p class="mut">Suggested tokens: trade + read scopes only (no payments, no admin). Refreshes every 10 s.</p>""",
                                  kv=kv, plans=plans)
    return _respond(_page("Capacity", body, refresh=10))


def register(app) -> None:
    app.register_blueprint(bp)
    if _ready():
        logger.info("FOLLOWER web pages enabled: /followers /verdict /capacity")
    else:
        logger.warning("FOLLOWER web pages refuse to load until ADMIN_PASSWORD and ENCRYPTION_KEY are set")
