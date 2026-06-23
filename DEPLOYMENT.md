# FastAPI AI Service - VPS Deployment Guide

This document provides step-by-step instructions on how to deploy the FastAPI AI service on a Virtual Private Server (VPS) using Docker and Docker Compose.

---

## Prerequisites

Before starting, ensure your VPS meets the following requirements:

- A Linux-based OS (Ubuntu 20.04/22.04 LTS recommended)
- Public IP address with SSH access
- A registered domain name pointing to your VPS IP (optional, but required for HTTPS/SSL)
- Incoming traffic allowed on ports `80` (HTTP), `443` (HTTPS), and `9676` (FastAPI direct port, optional)

---

## Option 1: Automated Deployment (Recommended)

An interactive deployment helper script `deploy-vps.sh` is provided in the repository root.

### Steps:

1. **Clone or Copy** this codebase onto your VPS.
2. **Navigate** to the project directory:
   ```bash
   cd /path/to/tarseah-ai-poc
   ```
3. **Execute** the deployment helper:
   ```bash
   ./deploy-vps.sh
   ```

### What the script does:

1. Checks for `docker` and installs it if missing.
2. Checks for `docker compose` and installs the plugin if missing.
3. Interactively prompts you to configure your API keys if a `.env` file does not exist.
4. Builds and starts the Docker container in detached mode.
5. Polls the application healthcheck endpoint to verify it started successfully.
6. Offers to clean up old dangling Docker images to preserve disk space.

---

## Option 2: Manual Deployment

If you prefer to configure the server manually, follow these steps:

### Step 1: Install Docker and Docker Compose

Install Docker on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

Ensure your user is added to the `docker` group so you don't need `sudo`:

```bash
sudo usermod -aG docker $USER
```

_(Log out and log back in to apply group changes)._

### Step 2: Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.examples .env
```

Open `.env` in a text editor (e.g., `nano .env`) and specify your production credentials:

```env
GOOGLE_API_KEY=your_production_google_gemini_api_key
GROQ_API_KEY=your_production_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
GROQ_MAX_OUTPUT_TOKENS=16384
```

### Step 3: Run the Docker Compose Stack

Build the image and launch the service:

```bash
docker compose up -d --build
```

### Step 4: Verify Deployment

Verify the containers are running:

```bash
docker compose ps
```

Check the healthcheck endpoint:

```bash
curl -f http://localhost:9676/
```

It should return a `200 OK` status with `{"message ": "Hello World !", "status": "ok"}`.

---

## Production Configurations

Running the FastAPI container directly on port 9676 is not recommended for production. It is best practice to run a reverse proxy (e.g., Nginx) in front of the container to handle SSL termination, rate limiting, and domain routing.

### Nginx Reverse Proxy Config

Create an Nginx server block configuration (e.g., `/etc/nginx/sites-available/ai.yourdomain.com`):

```nginx
server {
    listen 80;
    server_name ai.yourdomain.com;

    # Redirect all HTTP traffic to HTTPS
    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name ai.yourdomain.com;

    # SSL configuration (Certbot will fill this in automatically)
    # ssl_certificate /etc/letsencrypt/live/ai.yourdomain.com/fullchain.pem;
    # ssl_certificate_key /etc/letsencrypt/live/ai.yourdomain.com/privkey.pem;

    client_max_body_size 50M; # Allow large file uploads for RFP PDFs/Docs

    location / {
        proxy_pass http://127.0.0.1:9676;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Keep connection open for Server-Sent Events (SSE) streaming endpoints
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

Enable the site configuration and restart Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/ai.yourdomain.com /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### SSL Certificates (Let's Encrypt)

Obtain and configure free SSL certificates automatically using Certbot:

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d ai.yourdomain.com
```

---

## Operations and Maintenance

### Log Rotation & Viewing

Check application logs in real-time:

```bash
docker compose logs -f ai-service
```

To limit Docker log file sizes and avoid running out of disk space, configure Docker's log-driver in your VPS `/etc/docker/daemon.json` file:

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

_(Restart docker daemon after changes: `sudo systemctl restart docker`)_.

### Updating the Code

To pull updates and rebuild the containers without downtime:

```bash
git pull origin main
docker compose up -d --build --no-deps ai-service
```

### Data Storage & Persistence

All uploaded proposal documents and generated shared memory JSON states are stored in the docker volumes:

- `ai-storage` (Mapped internally to `/app/src/storage`)
- `ai-assets` (Mapped internally to `/app/src/Assets`)

You can locate the local storage directory of the volumes on the VPS by running:

```bash
docker volume inspect tarseah-ai-poc_ai-storage
```

---

## Troubleshooting

1. **Error: `502 Bad Gateway` from Nginx**
   - Ensure the Docker container is running: `docker compose ps`.
   - Verify that the port binding in `docker-compose.yml` matches what Nginx is proxying to (`127.0.0.1:9676`).

2. **API Failure: `Google API key not found` or `Groq API key not found`**
   - Ensure the `.env` file exists and is populated.
   - Restart the containers: `docker compose restart`.
   - Inspect container envs: `docker compose exec ai-service env | grep API_KEY`.

3. **Timeouts during LLM generation**
   - Generation endpoints can take up to 2-3 minutes for large proposals. Make sure Nginx `proxy_read_timeout` is set to `600s` and streaming routes are set up with `proxy_buffering off;`.

---

## Webhooks & Automated CI/CD Setup

To automate deployment whenever changes are pushed to your git repository, you can trigger the non-interactive `webhook-deploy.sh` script.

### Script Details: `webhook-deploy.sh`
This script executes:
1. `git fetch origin`
2. `git reset --hard origin/<branch>` (forces local repository state to match remote)
3. `docker compose up -d --build --remove-orphans`
4. Post-build health checks and error logging.

Logs are written with timestamps to `webhook-deploy.log`.

### Automated Hook Configuration Example (using `adnanh/webhook`)

1. **Install webhook tool** on your VPS:
   ```bash
   sudo apt-get install webhook
   ```

2. **Create configuration** (e.g., `/etc/webhook.conf`):
   ```json
   [
     {
       "id": "deploy-ai",
       "execute-command": "/path/to/tarseah-ai-poc/webhook-deploy.sh",
       "command-working-directory": "/path/to/tarseah-ai-poc",
       "response-message": "Deployment triggered successfully.",
       "trigger-rule": {
         "and": [
           {
             "match": {
               "type": "payload-hash-sha256",
               "secret": "YOUR_WEBHOOK_SECRET_TOKEN",
               "parameter": {
                 "source": "header",
                 "name": "X-Hub-Signature-256"
               }
             }
           },
           {
             "match": {
               "type": "value",
               "value": "refs/heads/main",
               "parameter": {
                 "source": "payload",
                 "name": "ref"
               }
             }
           }
         ]
       }
     }
   ]
   ```

3. **Configure GitHub/GitLab Webhook**:
   - **Payload URL**: `http://your-vps-ip:9000/hooks/deploy-ai`
   - **Content type**: `application/json`
   - **Secret**: `YOUR_WEBHOOK_SECRET_TOKEN`
   - **Events**: Just the `push` event.

4. **Run Webhook Service**:
   Run the webhook daemon:
   ```bash
   webhook -hooks /etc/webhook.conf -verbose -port 9000
   ```
   _(For production, configure it as a systemd service)._

