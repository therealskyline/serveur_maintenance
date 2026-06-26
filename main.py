#!/usr/bin/env python3
"""
main.py — Server coordinator for AnimeZone scraper.

Manages:
  - Distributed lock (active/passif) via HuggingFace
  - Heartbeat (keep-alive for active server)
  - Scrap scheduling (launch scrap.py, monitor, push results)
  - Checkpoints (push state.json to HF every 50 animes)
  - Failover (passive servers detect dead active and take over)

Usage:
  python main.py --hf <token>
  python main.py --hf <token> --repo skyline/animezone-catalog
  python main.py --hf <token> --server-id serv1 --port 10000

Deployment:
  - Render.com : set HF_TOKEN env var, deploy with Dockerfile
  - UptimeRobot : ping /health every 5 min to prevent spin-down
  - Multiple instances : deploy same code on N Render accounts
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import signal
import socket
import sqlite3
import subprocess
import sys
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
log = logging.getLogger("server")

DB_PATH = "animezone.db"
STATE_PATH = "state.json"
JSON_PATH = "animezone.json"
LOCK_FILE = "lock.json"
MANIFEST_FILE = "manifest.json"

HEARTBEAT_INTERVAL = 15
LOCK_TIMEOUT = 60
SCRAPE_INTERVAL = 6 * 3600
CHECKPOINT_INTERVAL = 50

app = FastAPI()
hf_api: HfApi | None = None
hf_token: str | None = None
hf_repo: str = ""
server_id: str = ""
is_active = False
scrap_process: subprocess.Popen | None = None
last_checkpoint_count = 0
shutdown_event = asyncio.Event()


@app.get("/health")
async def health():
    return {"status": "ok", "server_id": server_id, "is_active": is_active}


@app.get("/status")
async def status():
    scrap_running = scrap_process is not None and scrap_process.poll() is None
    return {
        "server_id": server_id,
        "is_active": is_active,
        "scrap_running": scrap_running,
        "hf_repo": hf_repo,
    }


async def hf_download(filename: str) -> str | None:
    try:
        return await asyncio.to_thread(
            hf_hub_download,
            repo_id=hf_repo,
            filename=filename,
            repo_type="dataset",
            token=hf_token,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    except Exception as e:
        log.debug("HF download %s : %s", filename, e)
        return None


async def hf_upload(local_path: str, remote_path: str) -> bool:
    try:
        await asyncio.to_thread(
            hf_api.upload_file,
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=hf_repo,
            repo_type="dataset",
            token=hf_token,
        )
        return True
    except Exception as e:
        log.error("HF upload %s : %s", remote_path, e)
        return False


async def ensure_hf_repo():
    try:
        await asyncio.to_thread(hf_api.repo_info, repo_id=hf_repo, repo_type="dataset")
    except Exception:
        log.info("Creation du repo HF %s ...", hf_repo)
        await asyncio.to_thread(
            hf_api.create_repo, repo_id=hf_repo, repo_type="dataset", private=True
        )
        log.info("✓ Repo cree")


async def fetch_lock() -> dict | None:
    path = await hf_download(LOCK_FILE)
    if path:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


async def push_lock(lock_data: dict):
    tmp = "/tmp/lock.json"
    with open(tmp, "w") as f:
        json.dump(lock_data, f)
    await hf_upload(tmp, LOCK_FILE)


async def try_acquire_lock() -> bool:
    global is_active
    lock = await fetch_lock()
    now = int(time.time())
    if lock and lock.get("last_heartbeat", 0) > now - LOCK_TIMEOUT:
        return False
    await asyncio.sleep(random.uniform(0, 3))
    lock = await fetch_lock()
    if lock and lock.get("last_heartbeat", 0) > now - LOCK_TIMEOUT:
        return False
    new_lock = {
        "active_server": server_id,
        "last_heartbeat": now,
        "lock_acquired_at": now,
        "scrap_in_progress": False,
    }
    await push_lock(new_lock)
    await asyncio.sleep(2)
    verify = await fetch_lock()
    if verify and verify.get("active_server") == server_id:
        is_active = True
        log.info("✓ Lock acquis — je suis le serveur actif")
        return True
    return False


async def update_heartbeat(scrap_info: dict | None = None) -> bool:
    global is_active
    lock = await fetch_lock()
    if not lock or lock.get("active_server") != server_id:
        if is_active:
            log.warning("Un autre serveur a pris le lock — je deviens passif")
        is_active = False
        return False
    lock["last_heartbeat"] = int(time.time())
    if scrap_info:
        lock["scrap_in_progress"] = True
        lock["scrap_progress"] = scrap_info
    else:
        lock["scrap_in_progress"] = False
        lock.pop("scrap_progress", None)
    await push_lock(lock)
    return True


async def download_existing_state():
    state_path = await hf_download(STATE_PATH)
    if state_path:
        os.replace(state_path, STATE_PATH)
        log.info("state.json telecharge depuis HF")
    else:
        log.info("Pas de state.json sur HF — demarrage from scratch")
    db_path = await hf_download(DB_PATH)
    if db_path:
        os.replace(db_path, DB_PATH)
        size_mb = os.path.getsize(DB_PATH) // (1024 * 1024)
        log.info("animezone.db telecharge depuis HF (%d Mo)", size_mb)
    else:
        log.info("Pas de DB sur HF — demarrage from scratch")


async def launch_scrap():
    global scrap_process, last_checkpoint_count
    try:
        with open(STATE_PATH) as f:
            state = json.load(f)
        last_checkpoint_count = len(state.get("animes_scraped", {}))
    except Exception:
        last_checkpoint_count = 0
    log.info("Lancement de scrap.py ...")
    scrap_process = subprocess.Popen(
        [sys.executable, "scrap.py", "--db", DB_PATH, "--state", STATE_PATH, "--json", JSON_PATH]
    )
    return scrap_process


async def get_scrap_progress() -> dict | None:
    try:
        with open(STATE_PATH) as f:
            state = json.load(f)
        count = len(state.get("animes_scraped", {}))
        animes = list(state.get("animes_scraped", {}).values())
        last_name = animes[-1].get("name", "?") if animes else "?"
        return {
            "animes_done": count,
            "current_anime": last_name,
            "started_at": state.get("last_incremental_scrape", 0),
        }
    except Exception:
        return None


async def monitor_scrap() -> bool:
    global last_checkpoint_count
    while scrap_process and scrap_process.poll() is None:
        if shutdown_event.is_set():
            scrap_process.terminate()
            break
        scrap_info = await get_scrap_progress()
        if not await update_heartbeat(scrap_info):
            log.warning("Lock perdu pendant le scrap — arret")
            scrap_process.terminate()
            return False
        try:
            with open(STATE_PATH) as f:
                state = json.load(f)
            current = len(state.get("animes_scraped", {}))
            if current - last_checkpoint_count >= CHECKPOINT_INTERVAL:
                await hf_upload(STATE_PATH, STATE_PATH)
                last_checkpoint_count = current
                log.info("Checkpoint : %d animes scrapes", current)
        except Exception:
            pass
        await asyncio.sleep(HEARTBEAT_INTERVAL)
    code = scrap_process.returncode if scrap_process else -1
    if code == 0:
        log.info("scrap.py termine avec succes")
        return True
    log.error("scrap.py crash (exit code %d)", code)
    return False


async def build_manifest() -> dict | None:
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    manifest = {
        "db_version": int(time.time()),
        "last_update": int(time.time()),
        "total_animes": c.execute("SELECT COUNT(*) FROM anime").fetchone()[0],
        "total_episodes": c.execute("SELECT COUNT(*) FROM episode").fetchone()[0],
        "total_urls": c.execute("SELECT COUNT(*) FROM episode_url").fetchone()[0],
        "db_size_bytes": os.path.getsize(DB_PATH),
    }
    sha = hashlib.sha256()
    with open(DB_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    manifest["db_sha256"] = sha.hexdigest()
    conn.close()
    return manifest


async def push_results_to_hf():
    if os.path.exists(DB_PATH):
        size_mb = os.path.getsize(DB_PATH) // (1024 * 1024)
        log.info("Push animezone.db (%d Mo) ...", size_mb)
        await hf_upload(DB_PATH, DB_PATH)
        log.info("✓ DB poussee")
    if os.path.exists(STATE_PATH):
        await hf_upload(STATE_PATH, STATE_PATH)
        log.info("✓ state.json pousse")
    manifest = await build_manifest()
    if manifest:
        tmp = "/tmp/manifest.json"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        await hf_upload(tmp, MANIFEST_FILE)
        log.info(
            "✓ Manifest : %d animes, %d episodes, %d URLs",
            manifest["total_animes"],
            manifest["total_episodes"],
            manifest["total_urls"],
        )


async def active_loop():
    global is_active
    is_active = True
    if not os.path.exists(STATE_PATH):
        await download_existing_state()
    await launch_scrap()
    success = await monitor_scrap()
    if success:
        await push_results_to_hf()
        log.info("Scrap termine — sleep %dh ...", SCRAPE_INTERVAL // 3600)
        sleep_start = time.time()
        while time.time() - sleep_start < SCRAPE_INTERVAL and not shutdown_event.is_set():
            if not await update_heartbeat():
                return
            await asyncio.sleep(HEARTBEAT_INTERVAL)
    else:
        log.error("Scrap echoue — retry dans 60s")
        await asyncio.sleep(60)


async def server_loop():
    global is_active
    await ensure_hf_repo()
    while not shutdown_event.is_set():
        lock = await fetch_lock()
        now = int(time.time())
        am_active = lock and lock.get("active_server") == server_id
        active_alive = lock and lock.get("last_heartbeat", 0) > now - LOCK_TIMEOUT
        if am_active:
            await active_loop()
        elif not active_alive:
            await try_acquire_lock()
        else:
            if is_active:
                is_active = False
            await asyncio.sleep(HEARTBEAT_INTERVAL)


def handle_shutdown(signum, frame):
    log.info("Signal d'arret recu")
    shutdown_event.set()


def main():
    global hf_api, hf_token, hf_repo, server_id
    parser = argparse.ArgumentParser(description="AnimeZone server coordinator")
    parser.add_argument("--hf", help="HuggingFace token (write access)")
    parser.add_argument("--repo", default="animezone-catalog", help="HF dataset repo ID")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", 10000)), help="Port"
    )
    parser.add_argument("--server-id", default=None, help="Unique server ID")
    args = parser.parse_args()

    hf_token = args.hf or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Erreur: fournir --hf <token> ou set HF_TOKEN env var")
        sys.exit(1)
    hf_repo = args.repo
    server_id = args.server_id or os.environ.get("SERVER_ID") or socket.gethostname()
    hf_api = HfApi(token=hf_token)

    log.info("=== AnimeZone Server ===")
    log.info("Server ID : %s", server_id)
    log.info("HF repo   : %s", hf_repo)
    log.info("Port      : %d", args.port)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    @app.on_event("startup")
    async def startup():
        asyncio.create_task(server_loop())

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
