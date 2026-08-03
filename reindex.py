"""
Offline reindex job. Run manually on a laptop, on a cron, or as a scheduled
GitHub Action - same script either way.

    python reindex.py            # incremental: only re-embeds changed pages
    python reindex.py --full     # re-embeds every page regardless of version

If STORAGE_MODE=git, commits the rebuilt chroma_db/ + bm25 index back to the
repo at the end, so ephemeral-disk hosts (Streamlit Community Cloud) pick up
the fresh index on their next auto-redeploy. If STORAGE_MODE=local, the index
is already durable on disk and nothing further is needed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import db
from config import settings
from confluence_client import ConfluenceClient
from parsing import chunk_page
from vectorstore import get_store


def run(full: bool = False) -> None:
    print(f"Reindex starting (mode={'full' if full else 'incremental'}, "
          f"storage_mode={settings.storage_mode}, space={settings.confluence_space_key})")

    db.init_db()
    client = ConfluenceClient()
    store = get_store()

    pages = client.list_pages()
    print(f"Fetched {len(pages)} pages from Confluence.")

    changed = 0
    skipped = 0
    total_chunks = 0

    for page in pages:
        if not full:
            synced_version = db.get_synced_version(page.page_id)
            if synced_version is not None and synced_version >= page.version:
                skipped += 1
                continue

        chunks = chunk_page(
            page_id=page.page_id,
            title=page.title,
            version=page.version,
            url=page.url,
            html=page.body_html,
        )
        store.delete_page(page.page_id)  # drop stale chunks before re-adding
        store.upsert_chunks(chunks)
        db.upsert_page_version(
            page_id=page.page_id, title=page.title, version=page.version, url=page.url
        )
        changed += 1
        total_chunks += len(chunks)
        print(f"  indexed: {page.title!r} (v{page.version}, {len(chunks)} chunks)")

    print(f"Done. {changed} pages (re)indexed, {skipped} unchanged, {total_chunks} chunks written.")

    if settings.storage_mode == "git" and changed > 0:
        _commit_index()


def _commit_index() -> None:
    """Commit the rebuilt index artifacts back to the repo. Only runs when
    STORAGE_MODE=git - i.e. only on hosts whose local disk doesn't persist."""
    paths = [settings.chroma_persist_dir, settings.bm25_index_path]
    try:
        subprocess.run(["git", "add", *paths], check=True)
        result = subprocess.run(
            ["git", "commit", "-m", "chore: scheduled reindex"], capture_output=True, text=True
        )
        if result.returncode != 0 and "nothing to commit" not in result.stdout:
            print(f"git commit warning: {result.stdout}{result.stderr}", file=sys.stderr)
            return
        subprocess.run(["git", "push"], check=True)
        print("Index committed and pushed.")
    except subprocess.CalledProcessError as e:
        print(f"git commit/push failed: {e}", file=sys.stderr)
        print("STORAGE_MODE=git requires this script to run somewhere with git "
              "credentials configured (e.g. a GitHub Action with GITHUB_TOKEN).",
              file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="re-embed every page, ignoring version cache")
    args = parser.parse_args()
    run(full=args.full)
