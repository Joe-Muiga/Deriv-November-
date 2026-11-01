"""analyze_rank_logs.py - how often do mass and edge ranking disagree, and is the board thinning?

Usage:  python analyze_rank_logs.py render.log [more.log ...]      (or pipe the log on stdin)
Reads the `RANK SHADOW window ...` lines (one per pick round) and the `WINDOW RANK ...: no admissible candidate` lines.
Also (when present): the `RANK OUTCOME ...` lines (theoretical outcomes of the top-N pool entries per pick round: hit rate vs
predicted p and ROI by rank / by trigger) and the `MH TRADE` lines (trades per hour over the time span of the log).
Logs without RANK OUTCOME lines / timestamps simply skip those sections.
"""
import collections, datetime, re, sys

SHADOW = re.compile(r"RANK SHADOW window (\S+) pool=(\d+) key=(\w+) mass_top=(\S+) edge_top=(\S+) agree=(\d)"
                    r".*?(?:trigger=(\w+))?\s*$")
NONE = re.compile(r"WINDOW RANK (\S+): no admissible candidate \| pool snapshot=(\d+) bar=(\d+) \| rejected: (.*)")
BOUGHT = re.compile(r"WINDOW RANK (\S+): bought (\d+) \| pool snapshot=(\d+) bar=(\d+)")
OUTCOME = re.compile(r"RANK OUTCOME win=(\S+) rank=(\d+) sym=(\S+) cover=(\S+) h=(\d+) trigger=(\w+) bought=([01]) "
                     r"p=([\d.]+) lb=([\d.]+) r=([\d.]+) hit=([01])")
STAMP = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")        # log prefix: %(asctime)s


def main(paths):
    lines = []
    for p in paths:
        with open(p, errors="replace") as f:
            lines += f.read().splitlines()
    if not paths:
        lines = sys.stdin.read().splitlines()
    rounds, agree, trig, pools = 0, 0, collections.Counter(), collections.Counter()
    by_trig = collections.defaultdict(lambda: [0, 0])
    none_n, none_rej = 0, collections.Counter()
    bought_n = 0
    outcomes = []                                           # (rank, trigger, bought, p, r, hit)
    trades, t_first, t_last = 0, None, None
    for ln in lines:
        ms = STAMP.match(ln)
        if ms:
            try:
                ts = datetime.datetime.strptime(ms.group(1).replace("T", " "), "%Y-%m-%d %H:%M:%S")
                t_first = ts if t_first is None or ts < t_first else t_first
                t_last = ts if t_last is None or ts > t_last else t_last
            except ValueError:
                pass
        if " MH TRADE " in ln:
            trades += 1
        mo = OUTCOME.search(ln)
        if mo:
            outcomes.append((int(mo.group(2)), mo.group(6), int(mo.group(7)), float(mo.group(8)), float(mo.group(10)), int(mo.group(11))))
            continue
        m = SHADOW.search(ln)
        if m:
            _w, pool, _k, _mt, _et, ag, tg = m.groups()
            rounds += 1
            agree += int(ag)
            tg = tg or "-"
            trig[tg] += 1
            by_trig[tg][0] += 1
            by_trig[tg][1] += int(ag)
            pools[min(int(pool), 10)] += 1
            continue
        m = NONE.search(ln)
        if m:
            none_n += 1
            for kv in m.group(4).split(","):
                if "=" in kv:
                    k, v = kv.strip().rsplit("=", 1)
                    none_rej[k] += int(v) if v.isdigit() else 0
            continue
        if BOUGHT.search(ln):
            bought_n += 1
    if not rounds:
        print("no RANK SHADOW lines found")
        _extra(outcomes, none_n, trades, t_first, t_last)
        return
    print(f"pick rounds: {rounds}")
    print(f"mass-top == edge-top: {agree}/{rounds} = {agree / rounds:.1%}   (disagree {rounds - agree} = {(rounds - agree) / rounds:.1%})")
    for k, (n, a) in sorted(by_trig.items()):
        print(f"  trigger={k:9s} rounds {n:5d}  agree {a / n:.1%}")
    print("pool size at pick (10 = 10+):", dict(sorted(pools.items())))
    early = trig.get("early", 0)
    print(f"early-trigger share: {early}/{rounds} = {early / rounds:.1%}   (low share = pool rarely reaches MH_EARLY_POOL_MIN)")
    tot_win = bought_n + none_n
    if tot_win:
        print(f"windows with a scan: {tot_win}, no admissible candidate: {none_n} = {none_n / tot_win:.1%}")
    if none_rej:
        print("rejections in empty windows:", dict(none_rej.most_common()))
    _extra(outcomes, none_n, trades, t_first, t_last)


def _roi(r, hit):
    return (r - 1.0) if hit else -1.0                      # theoretical profit per 1 staked


def _extra(outcomes, none_n, trades, t_first, t_last):
    """Sections added with the RANK OUTCOME lines / MH TRADE timestamps; every one is skipped when its lines are absent."""
    print(f"windows with no candidate: {none_n}")
    if outcomes:
        print(f"rank outcomes (theoretical: entry on the next tick, no slip / latency): {len(outcomes)}")
        by_rank = collections.defaultdict(list)
        by_trig = collections.defaultdict(list)
        for rank, trig, bought, p, r, hit in outcomes:
            by_rank[rank].append((p, r, hit))
            if bought:
                by_trig[trig].append((p, r, hit))
        print("  rank      n  hit rate  mean p   hit-p     ROI")
        for rank, v in sorted(by_rank.items()):
            n = len(v)
            hr, mp_ = sum(x[2] for x in v) / n, sum(x[0] for x in v) / n
            print(f"  {rank:4d} {n:6d}  {hr:8.1%}  {mp_:6.3f}  {hr - mp_:+6.3f}  {sum(_roi(x[1], x[2]) for x in v) / n:+7.3f}")
        if by_trig:
            print("ROI by trigger (picked entries only):")
            for trig, v in sorted(by_trig.items()):
                n = len(v)
                print(f"  trigger={trig:17s} n {n:5d}  hit {sum(x[2] for x in v) / n:6.1%}  ROI {sum(_roi(x[1], x[2]) for x in v) / n:+.3f}")
    else:
        print("no RANK OUTCOME lines found (old log format or MH_RANK_OUTCOME_TOPN=0)")
    if trades and t_first is not None and t_last is not None and t_last > t_first:
        span_h = (t_last - t_first).total_seconds() / 3600.0
        print(f"MH TRADE lines: {trades} over {span_h:.2f} h of log = {trades / span_h:.1f} trades/hour (the span includes idle time between sessions)")
    elif trades:
        print(f"MH TRADE lines: {trades} (no usable timestamps for a per-hour rate)")
    else:
        print("no MH TRADE lines found")


if __name__ == "__main__":
    main(sys.argv[1:])
