#!/bin/bash
#
# Setup SSH keys for video sync from worker to production server
# Run this script on each new worker server before starting containers
#

set -e

# Configuration
PROD_HOST="${VIDEO_SYNC_HOST:-51.79.65.199}"
PROD_USER="${VIDEO_SYNC_USER:-ubuntu}"
SSH_KEY_PATH="${HOME}/.ssh/id_ed25519"

echo "============================================"
echo "  Viralify Worker SSH Setup"
echo "============================================"
echo ""

# Step 1: Generate SSH key if not exists
if [ -f "$SSH_KEY_PATH" ]; then
    echo "[OK] SSH key already exists: $SSH_KEY_PATH"
else
    echo "[*] Generating new SSH key..."
    ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -C "viralify-worker@$(hostname)"
    echo "[OK] SSH key generated"
fi

echo ""

# Step 2: Display public key
echo "============================================"
echo "  PUBLIC KEY (add to production server)"
echo "============================================"
echo ""
cat "${SSH_KEY_PATH}.pub"
echo ""
echo "============================================"
echo ""
echo "To add this key to the production server, run:"
echo ""
echo "  ssh ${PROD_USER}@${PROD_HOST} \"echo '$(cat ${SSH_KEY_PATH}.pub)' >> ~/.ssh/authorized_keys\""
echo ""
echo "Or manually add the key above to: ${PROD_USER}@${PROD_HOST}:~/.ssh/authorized_keys"
echo ""

# Step 3: Test connection (if user wants)
read -p "Test SSH connection to ${PROD_HOST}? [y/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "[*] Testing SSH connection..."
    if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "${PROD_USER}@${PROD_HOST}" "echo 'SSH connection successful'" 2>/dev/null; then
        echo "[OK] SSH connection works!"

        # Verify target directory exists
        echo "[*] Verifying video sync path..."
        TARGET_PATH="${VIDEO_SYNC_PATH:-/var/lib/docker/volumes/repo_media_generator_videos/_data}"
        if ssh "${PROD_USER}@${PROD_HOST}" "sudo ls -la ${TARGET_PATH} > /dev/null 2>&1"; then
            echo "[OK] Target path exists: ${TARGET_PATH}"
        else
            echo "[WARN] Target path may not exist or is not accessible: ${TARGET_PATH}"
            echo "       Make sure the docker volume exists on the production server."
        fi
    else
        echo "[FAIL] SSH connection failed!"
        echo ""
        echo "Please ensure:"
        echo "  1. The public key is added to ${PROD_USER}@${PROD_HOST}:~/.ssh/authorized_keys"
        echo "  2. The production server is accessible from this worker"
        echo "  3. SSH is enabled on the production server"
        exit 1
    fi
fi

echo ""
echo "============================================"
echo "  Setup Complete"
echo "============================================"
echo ""
echo "Make sure .env.workers has:"
echo "  VIDEO_SYNC_ENABLED=true"
echo "  VIDEO_SYNC_HOST=${PROD_HOST}"
echo "  VIDEO_SYNC_USER=${PROD_USER}"
echo ""
echo "Then restart the presentation-generator container:"
echo "  docker compose -f docker-compose.workers.yml up -d --build presentation-generator"
echo ""
