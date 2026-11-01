"""Small tests for the follower rules. Run:  python test_follower_rules.py   (or pytest)."""
from datetime import datetime, timezone

import money_rules as mr
import verdict as vd
import zone_engine as ze
import stats_import

L = 5.0


def utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


# ── stake and floor math ─────────────────────────────────────────────────────
def test_growth_stake():
    s = mr.apply_reset(mr.FState(), 2.0, L, "d1")
    assert s.tier == "growth" and s.floor is None and s.p == 2.0
    assert mr.stake_for(s, 2.0) == 0.35            # 17.5% of 2.00 = 0.35
    assert mr.stake_for(s, 4.0) == 0.70
    assert mr.stake_for(s, 0.34) is None            # below MIN_STAKE: cannot trade
    assert mr.stake_for(s, 0.40) == 0.35            # never below MIN_STAKE, never above balance


def test_lock_floor_and_stake():
    s = mr.apply_reset(mr.FState(), 10.0, L, "d1")
    assert s.tier == "lock" and abs(s.floor - 7.0) < 1e-9          # 0.7 * P
    assert mr.stake_for(s, 10.0) == 0.52                           # 17.5% of 3.00 = 0.525 -> 0.52
    s = mr.update_on_balance(s, 16.0, L)                            # peak 16 -> P+0.5*(peak-P)=13
    assert abs(s.floor - 13.0) < 1e-9 and s.peak == 16.0
    assert mr.tradable(s, 16.0) == 3.0


def test_floor_not_pinned_to_p_at_reset():
    s = mr.apply_reset(mr.FState(), 10.0, L, "d1")
    s = mr.update_on_balance(s, 10.0, L)            # balance stream echo right after reset
    assert abs(s.floor - 7.0) < 1e-9 and mr.stake_for(s, 10.0) == 0.52
    s = mr.update_on_balance(s, 10.8, L)            # first gain: half of it is locked
    assert abs(s.floor - 10.4) < 1e-9


def test_growth_crosses_into_lock():
    s = mr.apply_reset(mr.FState(), 2.0, L, "d1")
    s = mr.update_on_balance(s, 4.9, L)
    assert s.floor is None                                          # still growth
    s = mr.update_on_balance(s, 6.0, L)                             # floor = max(1.4, 2+0.5*4)=4.0
    assert s.tier == "lock" and abs(s.floor - 4.0) < 1e-9


# ── buy rule ─────────────────────────────────────────────────────────────────
def test_buy_rule():
    s = mr.apply_reset(mr.FState(), 10.0, L, "d1")                  # floor 7
    assert mr.can_buy(s, 7.35, 0.35)                               # leaves exactly the floor
    assert not mr.can_buy(s, 7.34, 0.35)
    assert mr.stake_for(s, 7.34) is None and mr.is_stopped(s, 7.34)
    assert not mr.can_buy(s, 10.0, 0.34)                           # below MIN_STAKE
    assert mr.effective_balance(7.7, 0.35) == 7.7 - 0.35           # burst protection helper


def test_sticky_floor():
    s = mr.apply_reset(mr.FState(), 10.0, L, "d1")
    s = mr.update_on_balance(s, 20.0, L)                            # floor 15
    f = s.floor
    s = mr.update_on_balance(s, 4.0, L)                             # falls under L: floor stays
    assert s.floor == f and s.tier == "lock"
    s = mr.update_on_balance(s, 12.0, L)                            # still never moves down
    assert s.floor == f


# ── midnight reset at 21:00 UTC ──────────────────────────────────────────────
def test_reset_boundary():
    assert mr.kenya_day(utc(2026, 10, 11, 20, 59, 59)) == "2026-10-11"
    assert mr.kenya_day(utc(2026, 10, 11, 21, 0, 0)) == "2026-10-12"
    assert mr.kenya_day(utc(2026, 10, 11, 0, 0, 0)) == "2026-10-11"
    s = mr.apply_reset(mr.FState(), 2.0, L, "2026-10-11")
    assert not mr.needs_reset(s, mr.kenya_day(utc(2026, 10, 11, 20, 59)))
    assert mr.needs_reset(s, mr.kenya_day(utc(2026, 10, 11, 21, 0)))
    assert mr.needs_reset(mr.FState(), "2026-10-11")                # never reset yet


def test_reset_both_tiers():
    # lock tier, then reset below L -> floor dropped, growth tier, P/peak = live balance
    s = mr.update_on_balance(mr.apply_reset(mr.FState(), 10.0, L, "d1"), 20.0, L)
    s = mr.apply_reset(s, 3.0, L, "d2")
    assert s.tier == "growth" and s.floor is None and s.p == 3.0 and s.peak == 3.0
    # growth tier, then reset at/above L -> floor 0.7*P
    s = mr.apply_reset(s, 5.0, L, "d3")
    assert s.tier == "lock" and abs(s.floor - 3.5) < 1e-9
    assert abs(mr.milestone(s) - 5.0 * 1.325) < 1e-9
    assert abs(mr.milestone(mr.apply_reset(mr.FState(), 2.0, L, "d")) - 2.65) < 1e-9


# ── shrunk win rate and zone selection ───────────────────────────────────────
def test_shrunk_rate():
    assert abs(ze.shrunk_rate(60, 100, 0.4, k=200) - (60 + 80) / 300) < 1e-12
    assert abs(ze.shrunk_rate(0, 0, 0.4, k=200) - 0.4) < 1e-12


def _rows(per_minute=100, rate=0.40, ratio=2.42):
    return [[per_minute, int(per_minute * rate), per_minute * 0.35, per_minute * ratio, 0.0] for _ in range(60)]


def test_zone_selection():
    rows = _rows()
    assert ze.select_zones(rows, 150, 3) == []                     # 40% < break-even 41.3%: none qualify
    for m in range(10, 16):                                        # a genuinely strong block
        rows[m][1] = 55
    zones = ze.select_zones(rows, 150, 3)
    assert zones and all(z["ev"] > 0 and z["trades"] >= 150 for z in zones)
    mins = [set(range(z["start"], z["end"] + 1)) for z in zones]
    assert all(not (a & b) for i, a in enumerate(mins) for b in mins[i + 1:])
    assert 3 <= zones[0]["end"] - zones[0]["start"] + 1 <= 8
    assert ze.select_zones(rows, 10_000, 3) == []                  # min-trades gate


def test_plan_default_zone_and_tags():
    plan = ze.build_plan(_rows(), 53, 59)
    assert plan["tags"][53] == "default" and plan["tags"][59] == "default" and plan["tags"][52] is None
    assert plan["sessions"] == [{"start": 53, "end": 59}]
    rows = _rows()
    for m in (56, 57, 58, 59):                                      # strong block overlapping the default zone
        rows[m][1] = 60
    plan = ze.build_plan(rows, 53, 59)
    assert all(t != "engine" for t in plan["tags"][53:60])
    assert ze.sessions_from_tags([None] * 60) == []


# ── verdict states ───────────────────────────────────────────────────────────
def _obs(n, wins, ratio=2.42):
    return [((ratio - 1) if i < wins else -1.0, ratio) for i in range(n)]


def test_verdict_states():
    assert vd.summarize(_obs(50, 20), 300)["state"] == vd.NOT_ENOUGH
    assert vd.summarize(_obs(1000, 413), 300)["state"] == vd.CHANCE           # ~break-even
    assert vd.summarize(_obs(1000, 520), 300)["state"] == vd.PROFIT
    assert vd.summarize(_obs(1000, 300), 300)["state"] == vd.LOSS
    s = vd.summarize(_obs(1000, 300), 300)
    assert abs(s["break_even"] - 1 / 2.42) < 1e-9 and s["ci_high"] < 0


def test_verdict_one_row_per_signal_and_import():
    rows = [("default", -1.0, 2.42), ("engine", 1.42, 2.42)]
    out = vd.by_source(rows, 1)
    assert out["all"]["n"] == 2 and out["default"]["n"] == 1 and out["engine"]["n"] == 1
    raw = '{"v":1,"m":[' + ",".join(["[1,1,-0.05]"] * 60) + '],"t":[120,60,42.0,-3.0,290.4]}'
    conv = stats_import.convert(raw)
    assert len(conv) == 60 and conv[0][0] == 2 and abs(conv[0][4] - (-0.05 / 0.35)) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} tests passed")
