#!/usr/bin/env python3
"""
spawn_service.py - create the bot as NEW Render web service(s). One service, or a whole fleet in one go.

  pip install requests pyyaml

ONE SERVICE
  export RENDER_API_KEY=rnd_...
  python spawn_service.py my-bot-2 [--region frankfurt] [--stake 0.35] [--target 1] [--mode demo|real]
  (prompts for the Deriv token and app id)

A FLEET (e.g. 4 regions, 4 Render accounts/emails) - all created in parallel
  cp fleet.example.json fleet.json        # fill in the keys; fleet.json holds secrets: keep it OUT of git
  python spawn_service.py --fleet fleet.json [--dry-run]

Single source of truth = render.yaml (same folder): its `value:` env vars are copied to every service, and plan /
runtime / build + start commands are read from it. Per-service values come from flags, prompts or fleet.json:
  deriv_token (DERIV_API_TOKEN), deriv_app_id (DERIV_APP_ID), render_api_key (RENDER_API_KEY - each service gets the key
  of ITS OWN Render account), stake / target (initial PREDICTOR_STAKE / TARGET_PROFIT only: after creation this script
  never touches them, so what you type in that service's Render dashboard stays in charge), region, mode.
RENDER_SERVICE_ID is written automatically after each service is created, then the service is redeployed.
--dry-run prints each request (secrets masked) and creates nothing.
"""
import argparse
import getpass
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
import yaml

API = "https://api.render.com/v1"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STAKE = "0.35"
DEFAULT_TARGET = "1"
REGIONS = {"oregon", "ohio", "virginia", "frankfurt", "singapore"}
SECRET_KEYS = {"DERIV_API_TOKEN", "DERIV_APP_ID", "RENDER_API_KEY"}


class SpawnError(Exception):
    pass


def load_blueprint(path):
    with open(path, encoding="utf-8") as f:
        svc = yaml.safe_load(f)["services"][0]
    return {
        "shared": {e["key"]: str(e["value"]) for e in svc.get("envVars", []) if "value" in e},
        "runtime": svc.get("runtime", "python"),
        "plan": svc.get("plan", "free"),
        "region": svc.get("region", "oregon"),
        "build": svc.get("buildCommand", "pip install -r requirements.txt"),
        "start": svc.get("startCommand", "python main.py"),
    }


def git_repo_url():
    try:
        url = subprocess.check_output(["git", "-C", HERE, "remote", "get-url", "origin"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
    m = re.match(r"git@github\.com:(.+?)(?:\.git)?$", url)
    return f"https://github.com/{m.group(1)}" if m else re.sub(r"\.git$", "", url)


def call(key, method, path, **kw):
    h = {"Authorization": f"Bearer {key}", "Accept": "application/json", "Content-Type": "application/json"}
    r = requests.request(method, API + path, headers=h, timeout=30, **kw)
    if r.status_code >= 300:
        raise SpawnError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:300]}")
    return r.json() if r.text.strip() else {}


def pick_owner(key, wanted):
    if wanted:
        return wanted
    owners = [o["owner"] for o in call(key, "GET", "/owners", params={"limit": 20})]
    if len(owners) == 1:
        return owners[0]["id"]
    raise SpawnError("this Render account has several workspaces; set \"owner\" (tea-.../usr-...) for it: "
                     + ", ".join(f"{o['id']} ({o.get('name')})" for o in owners))


def mask(env):
    return [{"key": e["key"], "value": "****" if e["key"] in SECRET_KEYS else e["value"]} for e in env]


def build_payload(bp, repo, spec, owner):
    env = dict(bp["shared"])
    if spec.get("mode"):
        env["DERIV_ACCOUNT_MODE"] = spec["mode"]
    env.update({"DERIV_API_TOKEN": spec["deriv_token"], "DERIV_APP_ID": spec["deriv_app_id"],
                "RENDER_API_KEY": spec["render_api_key"],
                "PREDICTOR_STAKE": str(spec.get("stake", DEFAULT_STAKE)),
                "TARGET_PROFIT": str(spec.get("target", DEFAULT_TARGET))})
    return {
        "type": "web_service", "name": spec["name"], "ownerId": owner, "repo": repo,
        "branch": spec.get("branch", "main"), "autoDeploy": "yes",
        "envVars": [{"key": k, "value": v} for k, v in env.items()],
        "serviceDetails": {"runtime": bp["runtime"], "plan": bp["plan"], "region": spec["region"],
                           "envSpecificDetails": {"buildCommand": bp["build"], "startCommand": bp["start"]}},
    }


def create_one(bp, repo, spec, dry_run):
    """Create one service. Returns a result dict; never raises (so one failure can't stop the others)."""
    name, region = spec["name"], spec["region"]
    try:
        key = spec["render_api_key"]
        if dry_run:
            payload = build_payload(bp, repo, spec, spec.get("owner") or "<auto-detected workspace>")
            payload["envVars"] = mask(payload["envVars"])
            print(f"--- {name} ({region}) ---\n{json.dumps(payload, indent=2)}\n")
            return {"name": name, "region": region, "ok": True, "info": "dry run - nothing created"}
        owner = pick_owner(key, spec.get("owner"))
        for item in call(key, "GET", "/services", params={"name": name, "limit": 20}):
            if item["service"]["name"] == name:
                raise SpawnError(f"a service named '{name}' already exists in that account ({item['service']['id']})")
        created = call(key, "POST", "/services", json=build_payload(bp, repo, spec, owner))
        sid = created["service"]["id"]
        call(key, "PUT", f"/services/{sid}/env-vars/RENDER_SERVICE_ID", json={"value": sid})
        call(key, "POST", f"/services/{sid}/deploys", json={"clearCache": "do_not_clear"})
        url = created["service"].get("serviceDetails", {}).get("url", "")
        return {"name": name, "region": region, "ok": True, "id": sid,
                "info": f"https://dashboard.render.com/web/{sid}  {url}".strip()}
    except SpawnError as e:
        return {"name": name, "region": region, "ok": False, "info": str(e)}
    except Exception as e:  # network etc.
        return {"name": name, "region": region, "ok": False, "info": f"{type(e).__name__}: {e}"}


def load_fleet(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    shared, specs, problems = data.get("shared", {}), [], []
    for i, raw in enumerate(data.get("services", []), 1):
        s = {**shared, **raw}
        tag = s.get("name") or f"#{i}"
        for field in ("name", "region", "render_api_key", "deriv_token", "deriv_app_id"):
            if not str(s.get(field, "")).strip() or str(s.get(field)).startswith(("rnd_...", "PASTE")):
                problems.append(f"{tag}: '{field}' is missing")
        if s.get("region") and s["region"] not in REGIONS:
            problems.append(f"{tag}: region '{s['region']}' must be one of {sorted(REGIONS)}")
        if s.get("mode") and s["mode"] not in ("demo", "real"):
            problems.append(f"{tag}: mode must be demo or real")
        specs.append(s)
    names = [(s.get("render_api_key"), s.get("name")) for s in specs]
    if len(set(names)) != len(names):
        problems.append("two services share the same name AND the same Render account")
    if not specs:
        problems.append("no services listed")
    for field in ("deriv_token", "render_api_key"):
        seen = {}
        for s in specs:
            v = s.get(field)
            if v:
                seen.setdefault(v, []).append(s.get("name"))
        for v, who in seen.items():
            if len(who) > 1:
                print(f"NOTE: {', '.join(who)} share the same {field}.")
    return specs, problems


def warn_if_tracked(path):
    try:
        tracked = subprocess.run(["git", "-C", HERE, "check-ignore", "-q", os.path.abspath(path)]).returncode != 0
        inside = subprocess.run(["git", "-C", HERE, "rev-parse", "--is-inside-work-tree"],
                                capture_output=True).returncode == 0
        if inside and tracked:
            print(f"WARNING: {path} holds secrets and is not in .gitignore - add it before you commit anything.\n")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Create bot service(s) on Render.")
    ap.add_argument("name", nargs="?", help="service name (single-service mode)")
    ap.add_argument("--fleet", help="fleet.json: create every service in it, in parallel")
    ap.add_argument("--token", help="DERIV_API_TOKEN (prompted if omitted)")
    ap.add_argument("--app-id", help="DERIV_APP_ID (or $DERIV_APP_ID, or prompted)")
    ap.add_argument("--region", help="oregon | ohio | virginia | frankfurt | singapore (default: render.yaml)")
    ap.add_argument("--stake", default=DEFAULT_STAKE)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--mode", choices=["demo", "real"])
    ap.add_argument("--repo", help="GitHub repo URL (default: this folder's git origin)")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--owner", help="Render workspace id if the account has several")
    ap.add_argument("--blueprint", default=os.path.join(HERE, "render.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    bp = load_blueprint(a.blueprint)
    repo = a.repo or git_repo_url()
    if not repo:
        sys.exit("ERROR: could not detect the repo; pass --repo https://github.com/you/repo")

    if a.fleet:
        warn_if_tracked(a.fleet)
        specs, problems = load_fleet(a.fleet)
        if problems:
            print("Fix these in the fleet file first (nothing was created):")
            for p in problems:
                print("  -", p)
            sys.exit(1)
        for s in specs:
            s.setdefault("branch", a.branch)
        print(f"Creating {len(specs)} services in parallel ...\n")
        with ThreadPoolExecutor(max_workers=len(specs)) as ex:
            results = list(ex.map(lambda s: create_one(bp, repo, s, a.dry_run), specs))
    else:
        if not a.name:
            sys.exit("ERROR: give a service name, or use --fleet fleet.json")
        key = os.environ.get("RENDER_API_KEY", "").strip()
        if not key:
            sys.exit("ERROR: set RENDER_API_KEY in your environment first (Render -> Account Settings -> API Keys)")
        token = a.token or getpass.getpass("DERIV_API_TOKEN for this service (hidden): ").strip()
        app_id = a.app_id or os.environ.get("DERIV_APP_ID") or input("DERIV_APP_ID: ").strip()
        if not token or not app_id:
            sys.exit("ERROR: DERIV_API_TOKEN and DERIV_APP_ID are both required")
        spec = {"name": a.name, "region": a.region or bp["region"], "render_api_key": key, "deriv_token": token,
                "deriv_app_id": app_id, "stake": a.stake, "target": a.target, "mode": a.mode,
                "branch": a.branch, "owner": a.owner}
        if spec["region"] not in REGIONS:
            sys.exit(f"ERROR: region must be one of {sorted(REGIONS)}")
        results = [create_one(bp, repo, spec, a.dry_run)]

    print("\nRESULT")
    for r in results:
        print(f"  {'OK  ' if r['ok'] else 'FAIL'} {r['name']:<22} {r['region']:<10} {r['info']}")
    if any(not r["ok"] for r in results):
        print("\nRe-run just the failed ones: delete the OK entries from the fleet file first "
              "(a name that already exists in an account is refused, nothing is duplicated).")
        sys.exit(1)
    if not a.dry_run:
        print("\nChange PREDICTOR_STAKE / TARGET_PROFIT any time in each service's Render Environment tab.")


if __name__ == "__main__":
    main()
