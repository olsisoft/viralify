#!/usr/bin/env python3
"""
Auto-deployment webhook server for Viralify.

Listens for GitHub push events and automatically:
1. Pulls the latest code
2. Runs rebuild.sh
3. Sends notification when done

Usage:
    python3 deploy-webhook.py

Environment variables:
    WEBHOOK_SECRET - GitHub webhook secret (required)
    WEBHOOK_PORT - Port to listen on (default: 9000)
    DISCORD_WEBHOOK_URL - Discord webhook for notifications (optional)
    TELEGRAM_BOT_TOKEN - Telegram bot token (optional)
    TELEGRAM_CHAT_ID - Telegram chat ID (optional)
"""

import os
import sys
import hmac
import hashlib
import subprocess
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError

# Configuration
WEBHOOK_SECRET = os.environ.get('WEBHOOK_SECRET', 'your-secret-here')
WEBHOOK_PORT = int(os.environ.get('WEBHOOK_PORT', 9000))
REPO_PATH = os.environ.get('REPO_PATH', '/opt/viralify/repo')
REBUILD_SCRIPT = os.environ.get('REBUILD_SCRIPT', './rebuild.sh')

# Notification settings
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')


def log(message: str):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)


def send_notification(title: str, message: str, success: bool = True):
    """Send notification via configured channels."""
    color = 0x00FF00 if success else 0xFF0000  # Green or Red
    emoji = "✅" if success else "❌"

    # Discord notification
    if DISCORD_WEBHOOK_URL:
        try:
            payload = {
                "embeds": [{
                    "title": f"{emoji} {title}",
                    "description": message,
                    "color": color,
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }
            req = Request(
                DISCORD_WEBHOOK_URL,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            urlopen(req, timeout=10)
            log("Discord notification sent")
        except Exception as e:
            log(f"Discord notification failed: {e}")

    # Telegram notification
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            text = f"{emoji} *{title}*\n\n{message}"
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown"
            }
            req = Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            urlopen(req, timeout=10)
            log("Telegram notification sent")
        except Exception as e:
            log(f"Telegram notification failed: {e}")

    # Console notification (always)
    log(f"{'='*50}")
    log(f"{emoji} {title}")
    log(message)
    log(f"{'='*50}")


def verify_signature(payload: bytes, signature: str) -> bool:
    """Verify GitHub webhook signature."""
    if not signature:
        return False

    expected = 'sha256=' + hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


def run_deployment(commit_info: dict):
    """Run the deployment process."""
    commit_sha = commit_info.get('sha', 'unknown')[:7]
    commit_msg = commit_info.get('message', 'No message')
    author = commit_info.get('author', 'Unknown')

    log(f"Starting deployment for commit {commit_sha}")
    send_notification(
        "Déploiement en cours",
        f"**Commit:** `{commit_sha}`\n**Message:** {commit_msg}\n**Auteur:** {author}",
        success=True
    )

    start_time = time.time()

    try:
        # Change to repo directory
        os.chdir(REPO_PATH)

        # Git pull
        log("Running git pull...")
        result = subprocess.run(
            ['git', 'pull'],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            raise Exception(f"git pull failed: {result.stderr}")
        log(f"git pull: {result.stdout.strip()}")

        # Run rebuild script
        log(f"Running {REBUILD_SCRIPT}...")
        result = subprocess.run(
            ['bash', REBUILD_SCRIPT],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )

        elapsed = int(time.time() - start_time)

        if result.returncode != 0:
            raise Exception(f"Rebuild failed: {result.stderr[-500:]}")  # Last 500 chars

        # Success!
        send_notification(
            "Déploiement réussi! 🚀",
            f"**Commit:** `{commit_sha}`\n**Message:** {commit_msg}\n**Durée:** {elapsed}s\n\nL'application est prête!",
            success=True
        )

    except subprocess.TimeoutExpired:
        send_notification(
            "Déploiement échoué",
            f"**Commit:** `{commit_sha}`\n**Erreur:** Timeout - le script a pris trop de temps",
            success=False
        )
    except Exception as e:
        send_notification(
            "Déploiement échoué",
            f"**Commit:** `{commit_sha}`\n**Erreur:** {str(e)[:200]}",
            success=False
        )


class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP handler for GitHub webhooks."""

    def log_message(self, format, *args):
        """Override to use our logging."""
        log(f"HTTP: {args[0]}")

    def do_POST(self):
        """Handle POST requests (webhooks)."""
        if self.path != '/webhook':
            self.send_response(404)
            self.end_headers()
            return

        # Read payload
        content_length = int(self.headers.get('Content-Length', 0))
        payload = self.rfile.read(content_length)

        # Verify signature
        signature = self.headers.get('X-Hub-Signature-256', '')
        if WEBHOOK_SECRET != 'your-secret-here' and not verify_signature(payload, signature):
            log("Invalid webhook signature!")
            self.send_response(401)
            self.end_headers()
            return

        # Parse event
        event_type = self.headers.get('X-GitHub-Event', '')

        if event_type != 'push':
            log(f"Ignoring event type: {event_type}")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK - ignored')
            return

        # Parse payload
        try:
            data = json.loads(payload)

            # Get branch
            ref = data.get('ref', '')
            if ref != 'refs/heads/master' and ref != 'refs/heads/main':
                log(f"Ignoring push to branch: {ref}")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'OK - ignored branch')
                return

            # Get commit info
            head_commit = data.get('head_commit', {})
            commit_info = {
                'sha': head_commit.get('id', 'unknown'),
                'message': head_commit.get('message', 'No message').split('\n')[0],
                'author': head_commit.get('author', {}).get('name', 'Unknown')
            }

            # Respond immediately
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK - deployment started')

            # Run deployment in background thread
            thread = threading.Thread(target=run_deployment, args=(commit_info,))
            thread.daemon = True
            thread.start()

        except json.JSONDecodeError:
            log("Invalid JSON payload")
            self.send_response(400)
            self.end_headers()

    def do_GET(self):
        """Handle GET requests (health check)."""
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()


def main():
    """Start the webhook server."""
    log(f"Starting webhook server on port {WEBHOOK_PORT}")
    log(f"Repo path: {REPO_PATH}")
    log(f"Rebuild script: {REBUILD_SCRIPT}")
    log(f"Discord notifications: {'enabled' if DISCORD_WEBHOOK_URL else 'disabled'}")
    log(f"Telegram notifications: {'enabled' if TELEGRAM_BOT_TOKEN else 'disabled'}")

    if WEBHOOK_SECRET == 'your-secret-here':
        log("⚠️  WARNING: Using default webhook secret! Set WEBHOOK_SECRET env var.")

    server = HTTPServer(('0.0.0.0', WEBHOOK_PORT), WebhookHandler)

    try:
        log(f"Webhook server listening on http://0.0.0.0:{WEBHOOK_PORT}/webhook")
        server.serve_forever()
    except KeyboardInterrupt:
        log("Shutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
