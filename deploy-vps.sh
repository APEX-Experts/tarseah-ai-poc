#!/usr/bin/env bash

# Exit on error
set -e

# ANSI Color Codes for user-friendly output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}====================================================${NC}"
echo -e "${CYAN}     TARSEAH AI POC - VPS DEPLOYMENT HELPER         ${NC}"
echo -e "${CYAN}====================================================${NC}"

# Helper function to print messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Dependency Check: Docker
log_info "Checking system dependencies..."

if ! [ -x "$(command -v docker)" ]; then
    log_warning "Docker is not installed."
    read -p "Would you like to install Docker automatically? (y/n): " install_docker
    if [[ $install_docker =~ ^[Yy]$ ]]; then
        log_info "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        rm get-docker.sh
        sudo usermod -aG docker $USER
        log_success "Docker installed successfully! Please note you might need to log out and log back in for group changes to take effect."
    else
        log_error "Docker is required to proceed. Please install Docker and re-run this script."
        exit 1
    fi
else
    log_success "Docker is installed: $(docker --version)"
fi

# 2. Dependency Check: Docker Compose
DOCKER_COMPOSE_CMD=""
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    log_warning "Docker Compose is not installed."
    read -p "Would you like to install Docker Compose plugin? (y/n): " install_compose
    if [[ $install_compose =~ ^[Yy]$ ]]; then
        log_info "Installing Docker Compose plugin..."
        sudo apt-get update
        sudo apt-get install -y docker-compose-plugin
        DOCKER_COMPOSE_CMD="docker compose"
        log_success "Docker Compose installed successfully!"
    else
        log_error "Docker Compose is required to proceed. Please install docker-compose or the docker-compose-plugin and re-run this script."
        exit 1
    fi
fi
log_success "Docker Compose command found: $DOCKER_COMPOSE_CMD"

# 3. Environment File Configuration
if [ ! -f .env ]; then
    log_warning ".env file not found."
    if [ -f .env.examples ]; then
        log_info "Copying from .env.examples to create a new .env file..."
        cp .env.examples .env
    else
        log_info "Creating a fresh .env file..."
        touch .env
    fi

    echo -e "${YELLOW}--- Let's configure your environment variables ---${NC}"
    
    # Prompt for Google API Key
    read -p "Enter GOOGLE_API_KEY (leave blank to edit later): " google_key
    if [ ! -z "$google_key" ]; then
        # Check if GOOGLE_API_KEY exists in file to replace, else append
        if grep -q "GOOGLE_API_KEY" .env; then
            sed -i "s|GOOGLE_API_KEY=.*|GOOGLE_API_KEY=$google_key|" .env
        else
            echo "GOOGLE_API_KEY=$google_key" >> .env
        fi
    fi

    # Prompt for Groq API Key
    read -p "Enter GROQ_API_KEY (leave blank to edit later): " groq_key
    if [ ! -z "$groq_key" ]; then
        if grep -q "GROQ_API_KEY" .env; then
            sed -i "s|GROQ_API_KEY=.*|GROQ_API_KEY=$groq_key|" .env
        else
            echo "GROQ_API_KEY=$groq_key" >> .env
        fi
    fi

    # Prompt for Groq Model
    read -p "Enter GROQ_MODEL [default: openai/gpt-oss-20b]: " groq_model
    groq_model=${groq_model:-"openai/gpt-oss-20b"}
    if grep -q "GROQ_MODEL" .env; then
        sed -i "s|GROQ_MODEL=.*|GROQ_MODEL=$groq_model|" .env
    else
        echo "GROQ_MODEL=$groq_model" >> .env
    fi

    # Prompt for Groq Max Output Tokens
    read -p "Enter GROQ_MAX_OUTPUT_TOKENS [default: 16384]: " groq_tokens
    groq_tokens=${groq_tokens:-"16384"}
    if grep -q "GROQ_MAX_OUTPUT_TOKENS" .env; then
        sed -i "s|GROQ_MAX_OUTPUT_TOKENS=.*|GROQ_MAX_OUTPUT_TOKENS=$groq_tokens|" .env
    else
        echo "GROQ_MAX_OUTPUT_TOKENS=$groq_tokens" >> .env
    fi

    log_success ".env file configured successfully!"
else
    log_info ".env file already exists. Skipping interactive configuration."
fi

# 4. Start the Application using Docker Compose
log_info "Building and starting containers in detached mode..."
$DOCKER_COMPOSE_CMD up -d --build

# 5. Verify service health
log_info "Verifying service health..."
MAX_ATTEMPTS=12
ATTEMPT=1
HEALTHY=false

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    log_info "Checking health (attempt $ATTEMPT/$MAX_ATTEMPTS)..."
    # Port is 9676 based on docker-compose.yml port mapping
    if curl -s -f http://localhost:9676/ > /dev/null; then
        HEALTHY=true
        break
    fi
    sleep 5
    ATTEMPT=$((ATTEMPT+1))
done

if [ "$HEALTHY" = true ]; then
    log_success "Service is UP and HEALTHY!"
else
    log_error "Service failed to pass healthcheck within 60 seconds."
    log_info "Please run: '$DOCKER_COMPOSE_CMD logs' to inspect startup logs."
    exit 1
fi

# 6. Post-deployment Cleaning (Optional)
read -p "Would you like to prune dangling Docker images to clean up disk space? (y/n): " prune_images
if [[ $prune_images =~ ^[Yy]$ ]]; then
    log_info "Cleaning up unused Docker images/builders..."
    docker image prune -f
    log_success "Cleaned up dangling images!"
fi

echo -e "\n${GREEN}====================================================${NC}"
echo -e "${GREEN}             DEPLOYMENT SUCCESSFUL!                  ${NC}"
echo -e "${GREEN}====================================================${NC}"
echo -e "Your FastAPI AI service is running on port: ${CYAN}9676${NC}"
echo -e "\nUseful Commands:"
echo -e "  - View Logs:       ${CYAN}$DOCKER_COMPOSE_CMD logs -f${NC}"
echo -e "  - Restart Service: ${CYAN}$DOCKER_COMPOSE_CMD restart${NC}"
echo -e "  - Stop Service:    ${CYAN}$DOCKER_COMPOSE_CMD down${NC}"
echo -e "  - Check Status:    ${CYAN}$DOCKER_COMPOSE_CMD ps${NC}"
echo -e "====================================================\n"
