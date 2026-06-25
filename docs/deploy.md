# Deploying mai-telegram

This describes a **manual, self-hosted** deployment of mai-telegram to a remote
server with `make deploy`. The workflow assumes a specific (but common) topology:

- the bots run on a remote host inside an **LXD system container**, and
- inside that container the app runs as a **nested Docker Compose** stack
  (compose service `mai-gram`).

`make deploy` ships your local working tree to the server, takes a backup first,
rebuilds the image, and restarts the bots — **without touching** the production
config or the live chat database.

> If your topology is different (e.g. plain Docker on a bare host, no LXD), the
> ideas still apply but you'll want to adapt `scripts/deploy.sh`: the LXD
> snapshot/`lxc exec` parts are the LXD-specific bits.

## What is preserved (never overwritten / deleted)

These live only on the server and are excluded from the sync, so production
secrets and history are never clobbered by a deploy:

- `.env` — production bot tokens, proxy base URL/key, quiet hours, etc.
- `config/bots.toml` — your production bot roster (the repo only ships a small
  test config).
- `data/` — the SQLite DB (`mai_gram.db`), Chroma vectors, per-chat dirs, logs.

The in-container extraction uses a plain `tar -x` (no `--delete`), so any
server-only file inside the app directory survives a deploy.

## Prerequisites

- `ssh <your-host>` works and the host has passwordless `sudo` (for `lxc`).
- Docker + rsync available on the host / in the container.

## First-time setup

```bash
cp deploy.env.example deploy.env   # then edit the values for your server
```

`deploy.env` is gitignored — keep your server's real hostnames/paths out of the
repo. The variables:

| Variable | Example | Meaning |
|----------|---------|---------|
| `DEPLOY_SSH_HOST` | `my-server` | SSH alias or `user@host` for the server |
| `DEPLOY_LXD_CONTAINER` | `mai-telegram` | LXD container running the bots |
| `DEPLOY_APP_DIR` | `/opt/mai-telegram` | app dir inside the container |
| `DEPLOY_STAGING_DIR` | `.deploy-staging/mai-telegram` | rsync landing dir on the host |
| `DEPLOY_HDD_BACKUP_DIR` | `/var/backups/mai-telegram` | where `data/` tarballs land |
| `DEPLOY_KEEP_BACKUPS` | `5` | pre-deploy snapshots + tarballs to retain |
| `DEPLOY_SNAPSHOT_TIMEOUT` | `300` | max seconds to wait for the LXD snapshot |

## Deploy

```bash
make deploy
```

What it does, in order:

1. **Pre-flight** — checks SSH, that the container is `RUNNING`, warns on a dirty
   git tree.
2. **Backup** —
   - `lxc snapshot <container> pre-deploy-<UTC>` — best-effort full-container
     snapshot (captures code + DB + Docker state) for instant rollback. The btrfs
     copy itself is instant, but on some hosts recording the snapshot can be slow
     (see [below](#why-the-lxd-snapshot-can-be-slow)); if it exceeds
     `DEPLOY_SNAPSHOT_TIMEOUT` it is abandoned and the deploy continues. The
     deploy reports whether the snapshot was actually captured.
   - a tar of the whole `data/` (taken with `mai-gram` briefly stopped so the
     SQLite DB + Chroma are quiescent), pulled to
     `$DEPLOY_HDD_BACKUP_DIR/data-<UTC>.tar.gz` with a `.sha256`. This is the
     **primary, guaranteed** rollback artifact for the chat history.
   - prunes old pre-deploy snapshots and tarballs to `DEPLOY_KEEP_BACKUPS`.
3. **Sync** — rsync the working tree to the host staging dir, then tar-pipe it
   into the container (server-only files preserved).
4. **Rebuild + restart** — `docker compose build && docker compose up -d`.
5. **Verify** — confirms `mai-gram` is `running` and scans the last log lines for
   `getUpdates` conflicts, `Unauthorized`, a missing API key, or tracebacks;
   fails loudly (with a rollback hint) if anything looks wrong.

## Verify manually

```bash
ssh <your-host> "sudo lxc exec <container> -- docker logs mai-gram --tail 40"
ssh <your-host> "sudo lxc exec <container> -- sh -c 'cd <app-dir> && docker compose ps'"
```

Then send a message to one of the bots and confirm it replies.

## Rollback

The deploy prints the exact commands on completion. In short:

**Instant full rollback** (code + DB + Docker state) — only if a snapshot was
captured this run:

```bash
ssh <your-host> 'sudo lxc stop <container> && \
    sudo lxc restore <container> pre-deploy-<UTC> && \
    sudo lxc start <container>'
```

**Restore chat history only** from the host tarball (always available):

```bash
ssh <your-host> 'sudo lxc file push <backup-dir>/data-<UTC>.tar.gz <container>/tmp/ && \
    sudo lxc exec <container> -- tar xzf /tmp/data-<UTC>.tar.gz -C <app-dir>'
```

List available restore points:

```bash
ssh <your-host> "sudo lxc info <container> | sed -n '/Snapshots:/,\$p'"
ssh <your-host> "ls -lh <backup-dir>/"
```

## How the backup works (and what the tarball is)

There are **two** independent pre-deploy artifacts, with very different cost and
guarantees:

1. **The `data/` tarball** — a `tar czf data-<UTC>.tar.gz` of the `data/`
   directory: the SQLite chat DB, Chroma vectors and per-chat dirs. To keep
   SQLite + Chroma internally consistent it's taken with `mai-gram` briefly
   stopped (a few seconds), then pulled off the container to the host with a
   `sha256` checksum. `data/` is small (typically tens of MB), so this is **fast
   and always succeeds**. It is the primary rollback artifact for the chat
   history.

2. **The LXD snapshot** — a whole-container point-in-time image (code + DB +
   Docker layers). It's a btrfs copy-on-write snapshot, so it captures the
   *entire* container (often a few GB, mostly Docker images/build cache) without
   copying the data. It enables a one-command rollback of everything, not just
   `data/`. It's **best-effort** because it can be slow on a contended daemon.

### Why the LXD snapshot can be slow (and why the deploy submits it async)

The btrfs snapshot itself is **instant** (copy-on-write — no data is copied), and
the daemon-side operation typically finishes in ~1-2s. The trap is the *client*:
a foreground `lxc snapshot` **blocks waiting for the operation to complete**, and
on a contended LXD daemon that wait can hang for *minutes* even though the
snapshot is already done — and interrupting the hung client (e.g. with `timeout`)
can abort the operation, leaving no snapshot.

So the deploy doesn't run a blocking `lxc snapshot`. It **submits the snapshot as
an async daemon operation via the REST API** (`lxc query -X POST
/1.0/instances/<c>/snapshots`), which returns immediately, then **polls** for the
snapshot to appear (`lxc info`). The daemon owns the operation, so a slow client
can't stall the deploy and can't abort the snapshot. `DEPLOY_SNAPSHOT_TIMEOUT`
just bounds how long we poll before continuing best-effort.

If snapshots are *persistently* slow on your host even via the async path, the
daemon itself is likely degraded (e.g. a backlog of stuck operations after an
out-of-disk incident). The clean fix is to restart the LXD daemon
(`snap restart lxd`) to clear the backlog — but be aware that on some setups this
**stops running containers** and tears down custom bridges/routes, so wrap it in
a script that restarts the containers and re-applies any networking afterward
rather than running it bare.

## Notes

- This per-deploy backup is meant to *complement*, not replace, whatever
  scheduled host-level backups you run (e.g. periodic LXD auto-snapshots and
  off-box exports). Because mai-telegram's SQLite DB is a host-visible bind mount
  under the app dir's `data/`, a container snapshot/export already captures it —
  no separate database-dump job is required.
