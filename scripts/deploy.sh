#!/usr/bin/env bash
# Manual deploy of mai-telegram to a remote server.
#
# This targets a setup where mai-telegram runs inside an LXD system container on
# the remote host, as a nested Docker Compose stack (compose service `mai-gram`).
# The deploy is a two-hop sync:
#   1. rsync the local working tree to a staging dir on the remote host
#   2. tar-pipe the staging dir into the container's app directory
#   3. rebuild + restart the Docker Compose stack inside the container
#
# Before any of that, a pre-deploy backup is taken:
#   - a best-effort LXD (btrfs CoW) snapshot of the whole container, AND
#   - a consistent tar of data/ pulled to a backup directory on the host
#     (with a .sha256).
# This protects the live chat history during an upgrade and gives a rollback
# path if a deploy goes wrong.
#
# Production-only files are NEVER shipped and NEVER deleted on the server:
#   - rsync excludes .env, config/bots.toml(.old), data/, deploy.env, caches
#   - the in-container extraction uses plain `tar -x` (no --delete), so files
#     that only exist on the server stay put.
#
# All environment-specific values come from the gitignored `deploy.env` in the
# repo root. Copy `deploy.env.example` to `deploy.env` and fill it in.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$REPO_ROOT/deploy.env"

log()  { printf '\033[36m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[deploy] WARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[deploy] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ -f "$CONFIG_FILE" ]] || die "deploy.env not found. Copy deploy.env.example to deploy.env and fill it in."

# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${DEPLOY_SSH_HOST:?DEPLOY_SSH_HOST must be set in deploy.env}"
: "${DEPLOY_LXD_CONTAINER:?DEPLOY_LXD_CONTAINER must be set in deploy.env}"
: "${DEPLOY_APP_DIR:?DEPLOY_APP_DIR must be set in deploy.env}"
DEPLOY_STAGING_DIR="${DEPLOY_STAGING_DIR:-.deploy-staging/mai-telegram}"
DEPLOY_KEEP_BACKUPS="${DEPLOY_KEEP_BACKUPS:-5}"
DEPLOY_ALLOW_DIRTY="${DEPLOY_ALLOW_DIRTY:-0}"
DEPLOY_HDD_BACKUP_DIR="${DEPLOY_HDD_BACKUP_DIR:-/var/backups/mai-telegram}"
# Max seconds to poll for the LXD snapshot to appear. The snapshot is submitted as
# an async daemon operation (see below) which normally completes in ~1-2s; this
# only bounds how long we wait before giving up and continuing best-effort (the
# portable data/ tarball is the primary rollback artifact). See docs/deploy.md.
DEPLOY_SNAPSHOT_TIMEOUT="${DEPLOY_SNAPSHOT_TIMEOUT:-120}"

# Compose service name inside the container (must match docker-compose.yml).
COMPOSE_SERVICE="mai-gram"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
SNAPSHOT="pre-deploy-$TS"
SNAP_OK=0   # set to 1 once the LXD snapshot is confirmed to exist

# Helper: run a shell command on the remote host.
on_host() { ssh "$DEPLOY_SSH_HOST" "$1"; }

# Helper: run a shell command inside the container's app dir.
in_app() {
    ssh "$DEPLOY_SSH_HOST" \
        "sudo lxc exec '$DEPLOY_LXD_CONTAINER' -- sh -c 'cd \"$DEPLOY_APP_DIR\" && $1'"
}

print_rollback_hint() {
    {
        echo
        echo "────────────────────────────────────────────────────────────────────────"
        if [ "${SNAP_OK:-0}" = "1" ]; then
            echo "Rollback options (snapshot: $SNAPSHOT):"
            echo
            echo "  Instant full rollback (code + DB + Docker state):"
            echo "    ssh $DEPLOY_SSH_HOST 'sudo lxc stop $DEPLOY_LXD_CONTAINER && \\"
            echo "        sudo lxc restore $DEPLOY_LXD_CONTAINER $SNAPSHOT && \\"
            echo "        sudo lxc start $DEPLOY_LXD_CONTAINER'"
            echo
        else
            echo "Rollback options (no LXD snapshot was captured this run):"
            echo
        fi
        echo "  Restore data/ only (chat history) from the HDD tarball:"
        echo "    ssh $DEPLOY_SSH_HOST 'sudo lxc file push \\"
        echo "        $DEPLOY_HDD_BACKUP_DIR/data-$TS.tar.gz $DEPLOY_LXD_CONTAINER/tmp/ && \\"
        echo "        sudo lxc exec $DEPLOY_LXD_CONTAINER -- tar xzf /tmp/data-$TS.tar.gz -C $DEPLOY_APP_DIR'"
        echo "────────────────────────────────────────────────────────────────────────"
    } >&2
}

# ── Pre-flight ────────────────────────────────────────────────────────────
log "Pre-flight checks ..."

ssh -o ConnectTimeout=10 "$DEPLOY_SSH_HOST" true \
    || die "Cannot SSH to '$DEPLOY_SSH_HOST'."

state="$(on_host "sudo lxc list '$DEPLOY_LXD_CONTAINER' -c s -f csv" | head -n1)"
[[ "$state" == "RUNNING" ]] \
    || die "Container '$DEPLOY_LXD_CONTAINER' is not RUNNING (state: ${state:-unknown})."

if [[ "$DEPLOY_ALLOW_DIRTY" != "1" ]]; then
    if ! git -C "$REPO_ROOT" diff --quiet HEAD 2>/dev/null \
        || [[ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]]; then
        warn "Local git tree is dirty (uncommitted changes will be deployed)."
        warn "Set DEPLOY_ALLOW_DIRTY=1 in deploy.env to silence this, or commit first."
    fi
fi

# ── Pre-deploy backup ───────────────────────────────────────────────────────
# The LXD snapshot is taken FIRST, while the stack is still running. It is a
# crash-consistent btrfs CoW image (equivalent to a power-cut), which SQLite
# recovers from cleanly via its journal/WAL — so it is a safe full rollback.
#
# We submit it as an async daemon operation via the REST API (`lxc query`) and
# then poll for the snapshot to appear, rather than running a blocking
# `lxc snapshot`. The btrfs CoW copy itself is instant; the reason for this
# indirection is that the foreground `lxc snapshot` *client* blocks waiting for
# the operation to complete, and on a contended LXD daemon that wait can hang for
# minutes even though the snapshot is already done — and interrupting that client
# can abort the operation. Submitting async sidesteps both: the daemon owns the
# operation, so we just watch for the result. When healthy this returns in ~1-2s.
log "Taking LXD snapshot '$SNAPSHOT' (async submit + poll, up to ${DEPLOY_SNAPSHOT_TIMEOUT}s) ..."
if on_host "
    sudo lxc query -X POST '/1.0/instances/$DEPLOY_LXD_CONTAINER/snapshots' \
        -d '{\"name\":\"$SNAPSHOT\"}' >/dev/null 2>&1 || true
    for _ in \$(seq 1 $(( (DEPLOY_SNAPSHOT_TIMEOUT + 1) / 2 ))); do
        if sudo lxc info '$DEPLOY_LXD_CONTAINER' 2>/dev/null | grep -q '$SNAPSHOT'; then
            exit 0
        fi
        sleep 2
    done
    exit 1
"; then
    SNAP_OK=1
    log "Snapshot created."
else
    warn "LXD snapshot did not appear within ${DEPLOY_SNAPSHOT_TIMEOUT}s (daemon contended; see docs/deploy.md)."
    warn "Continuing WITHOUT an LXD snapshot. The portable data/ tarball below is the rollback artifact for the chat history."
fi

# For the portable HDD tarball we want a clean, quiescent copy. sqlite3 is not
# installed in the container, so instead of an online .backup we briefly stop
# the bots, tar data/ while nothing is writing, and leave the stack stopped —
# the later `docker compose up -d` brings it back. This keeps the chat DB
# (mai_gram.db), Chroma vectors and per-chat dirs internally consistent.
log "Stopping '$COMPOSE_SERVICE' for a consistent data backup ..."
in_app "docker compose stop $COMPOSE_SERVICE"

log "Backing up data/ to $DEPLOY_HDD_BACKUP_DIR ..."
on_host "sudo mkdir -p '$DEPLOY_HDD_BACKUP_DIR'"
in_app "tar czf /tmp/data-$TS.tar.gz -C '$DEPLOY_APP_DIR' data/"

on_host "
    set -e
    sudo lxc file pull '$DEPLOY_LXD_CONTAINER/tmp/data-$TS.tar.gz' '$DEPLOY_HDD_BACKUP_DIR/data-$TS.tar.gz'
    sudo lxc exec '$DEPLOY_LXD_CONTAINER' -- rm -f '/tmp/data-$TS.tar.gz'
    cd '$DEPLOY_HDD_BACKUP_DIR'
    sudo sha256sum 'data-$TS.tar.gz' | sudo tee 'data-$TS.tar.gz.sha256' >/dev/null
    ls -lh 'data-$TS.tar.gz'
"

# Prune old pre-deploy snapshots and HDD tarballs (keep newest N).
log "Pruning old backups (keeping newest $DEPLOY_KEEP_BACKUPS) ..."
on_host "
    snaps=\$(sudo lxc info '$DEPLOY_LXD_CONTAINER' | sed -n '/Snapshots:/,\$p' \
        | grep -oE 'pre-deploy-[0-9TZ]+' | sort)
    total=\$(printf '%s\n' \$snaps | grep -c . || true)
    if [ \"\$total\" -gt $DEPLOY_KEEP_BACKUPS ]; then
        printf '%s\n' \$snaps | head -n \$((total - $DEPLOY_KEEP_BACKUPS)) | while read -r s; do
            [ -n \"\$s\" ] && sudo lxc delete '$DEPLOY_LXD_CONTAINER/'\"\$s\"
        done
    fi
"
on_host "
    cd '$DEPLOY_HDD_BACKUP_DIR' 2>/dev/null || exit 0
    ls -1t data-*.tar.gz 2>/dev/null | tail -n +\$(($DEPLOY_KEEP_BACKUPS + 1)) | while read -r f; do
        sudo rm -f \"\$f\" \"\$f.sha256\"
    done
"

# ── Sync code ───────────────────────────────────────────────────────────────
RSYNC_EXCLUDES=(
    --include='.env.example'
    --exclude='.git/'
    --exclude='.env'
    --exclude='.env.*'
    --exclude='deploy.env'
    --exclude='.deploy-staging/'
    --exclude='config/bots.toml'
    --exclude='config/bots.toml.old'
    --exclude='data/'
    --exclude='backups/'
    --exclude='avatars/'
    --exclude='__pycache__/'
    --exclude='.venv/'
    --exclude='venv/'
    --exclude='.mypy_cache/'
    --exclude='.ruff_cache/'
    --exclude='.pytest_cache/'
    --exclude='.coverage'
    --exclude='htmlcov/'
    --exclude='*.log'
    --exclude='.cursor/'
    --exclude='.idea/'
    --exclude='.vscode/'
    --exclude='tmp/'
)

log "Syncing working tree to $DEPLOY_SSH_HOST:$DEPLOY_STAGING_DIR ..."
on_host "mkdir -p '$DEPLOY_STAGING_DIR'"
rsync -az --delete "${RSYNC_EXCLUDES[@]}" "$REPO_ROOT/" "$DEPLOY_SSH_HOST:$DEPLOY_STAGING_DIR/"

log "Copying code into '$DEPLOY_LXD_CONTAINER' at $DEPLOY_APP_DIR (server-only files preserved) ..."
on_host "tar -C '$DEPLOY_STAGING_DIR' -cf - . | sudo lxc exec '$DEPLOY_LXD_CONTAINER' -- tar -C '$DEPLOY_APP_DIR' -xof -"

# ── Rebuild + restart ───────────────────────────────────────────────────────
trap 'die "Deploy failed. See errors above."; print_rollback_hint' ERR

log "Building image ..."
in_app "docker compose build"

log "Restarting stack ..."
in_app "docker compose up -d"

log "Stack status:"
in_app "docker compose ps"

# ── Verify ──────────────────────────────────────────────────────────────────
trap - ERR
# Poll for the service to reach 'running'. Right after a recreate the State field
# is briefly empty, so a single short sleep races; retry for ~30s instead.
log "Waiting for '$COMPOSE_SERVICE' to come up ..."
state=""
for _ in $(seq 1 15); do
    state="$(in_app "docker compose ps --format '{{.State}}' $COMPOSE_SERVICE" 2>/dev/null | tr -d '[:space:]' || true)"
    [ "$state" = "running" ] && break
    sleep 2
done
in_app "docker compose ps"
if [ "$state" != "running" ]; then
    warn "Service '$COMPOSE_SERVICE' is not 'running' after ~30s (state: ${state:-unknown})."
    print_rollback_hint
    die "Post-deploy check failed."
fi

log "Service is running. Recent logs:"
logs="$(in_app "docker compose logs --tail 40 $COMPOSE_SERVICE" 2>&1 || true)"
echo "$logs"
if grep -qiE 'terminated by other getUpdates|Unauthorized|invalid token|OPENROUTER_API_KEY|Traceback' <<<"$logs"; then
    warn "Found suspicious lines in the logs above."
    print_rollback_hint
    die "Post-deploy log check failed."
fi

if [ "$SNAP_OK" = "1" ]; then
    log "Deploy complete. Snapshot: $SNAPSHOT  |  HDD backup: $DEPLOY_HDD_BACKUP_DIR/data-$TS.tar.gz"
else
    log "Deploy complete (no LXD snapshot).  HDD backup: $DEPLOY_HDD_BACKUP_DIR/data-$TS.tar.gz"
fi
print_rollback_hint
