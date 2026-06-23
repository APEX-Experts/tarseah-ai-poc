#!/usr/bin/env bash

# Non-interactive Git pull and Docker Compose deployment script
# Designed to be triggered by webhooks, crontabs, or CI/CD pipelines on the VPS.

set -euo pipefail

# Configurations
BRANCH="${1:-main}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${PROJECT_DIR}/webhook-deploy.log"

# Function to log messages with timestamps
log() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$message"
    echo -e "$message" >> "$LOG_FILE"
}

# Redirect stderr to log file as well
exec 2> >(while read -r line; do log "[ERROR] $line"; done)

log "===================================================="
log "Starting automated deployment for branch: ${BRANCH}"
log "===================================================="

# Navigate to project directory
cd "$PROJECT_DIR"

# 1. Fetch latest changes from remote
log "Fetching latest changes from origin..."
git fetch origin

# 2. Force checkout and reset to match origin branch
log "Resetting local codebase to origin/${BRANCH}..."
git checkout -B "$BRANCH" "origin/$BRANCH"
git reset --hard "origin/$BRANCH"

# 3. Identify docker compose command
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    log "[ERROR] Docker Compose is not installed on the system."
    exit 1
fi

# 4. Pull any updated base images, build and start container
log "Rebuilding and restarting Docker containers..."
$DOCKER_COMPOSE_CMD up -d --build --remove-orphans

# 5. Verify service health
log "Verifying service health..."
MAX_ATTEMPTS=12
ATTEMPT=1
HEALTHY=false

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    log "Checking health (attempt $ATTEMPT/$MAX_ATTEMPTS)..."
    # Port is 9676 based on docker-compose.yml port mapping
    if curl -s -f http://localhost:9676/ > /dev/null; then
        HEALTHY=true
        break
    fi
    sleep 5
    ATTEMPT=$((ATTEMPT+1))
done

if [ "$HEALTHY" = true ]; then
    log "Automated deployment completed SUCCESSFULLY! Service is online."
else
    log "[CRITICAL] Deployment failed health check. Inspecting logs..."
    log "--- Last 50 lines of Docker logs ---"
    $DOCKER_COMPOSE_CMD logs --tail=50 >> "$LOG_FILE"
    exit 1
fi

log "===================================================="
log "Deployment finished successfully."
log "===================================================="
