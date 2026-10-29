"""Fetch the repo's GitHub issues for the project-management page.

AI-assisted (Databricks Genie Code), issue #184.

Reads every issue (pull requests excluded) from the GitHub REST API and writes
the fields the page needs to site/data/pm/issues.json. Only public issue
metadata is used: number, title, state, dates, labels and people.

The owner of an issue is its first assignee or, if nobody is assigned, its
author. GitHub logins are mapped to first names.

Usage:  GITHUB_TOKEN=... python site/scripts/build_pm.py [--repo owner/name] [--out FILE]
The token is optional for a public repo (it raises the API rate limit).
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PEOPLE = {
    "almac09": "Alan",
    "bridgetmacdonnell1-dev": "Bridget",
    "CianHerreraQuinn": "Cian",
    "gerardmurphh": "Gerard",
}


def get(url: str, token: str | None):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "iaa-hackathon-site",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r), r.headers.get("Link", "")


def next_link(link: str) -> str | None:
    for part in link.split(","):
        if 'rel="next"' in part:
            return part[part.index("<") + 1: part.index(">")]
    return None


def name(login: str | None) -> str:
    return PEOPLE.get(login or "", login or "unknown")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "almac09/iaa-hackathon-2026"))
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "site/data/pm/issues.json")
    a = ap.parse_args()
    token = os.environ.get("GITHUB_TOKEN")

    url = f"https://api.github.com/repos/{a.repo}/issues?state=all&per_page=100&sort=created&direction=asc"
    rows = []
    while url:
        page, link = get(url, token)
        for i in page:
            if "pull_request" in i:
                continue
            assignees = [x["login"] for x in i.get("assignees") or []]
            owner_login = assignees[0] if assignees else (i.get("user") or {}).get("login")
            rows.append({
                "number": i["number"],
                "title": i["title"],
                "state": i["state"],
                "state_reason": i.get("state_reason"),
                "created_at": i["created_at"],
                "closed_at": i.get("closed_at"),
                "labels": [l["name"] for l in i.get("labels") or []],
                "author": name((i.get("user") or {}).get("login")),
                "owner": name(owner_login),
                "url": i["html_url"],
            })
        url = next_link(link)

    if not rows:
        raise SystemExit("no issues returned; refusing to publish an empty page")
    out = {
        "repo": a.repo,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issues": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    closed = sum(r["state"] == "closed" for r in rows)
    print(f"wrote {a.out}: {len(rows)} issues, {closed} closed")


if __name__ == "__main__":
    main()
