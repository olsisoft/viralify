#!/bin/bash
# =============================================================================
# Run SQL Migration Script
# =============================================================================
# Usage: ./run_migration.sh <migration_file>
# Example: ./run_migration.sh 002_weave_graph.sql
# =============================================================================

set -e

# Configuration
CONTAINER_NAME="${POSTGRES_CONTAINER:-tiktok-postgres}"
DB_USER="${POSTGRES_USER:-tiktok_user}"
DB_NAME="${POSTGRES_DB:-tiktok_platform}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: $0 <migration_file>${NC}"
    echo ""
    echo "Available migrations:"
    ls -1 *.sql 2>/dev/null || echo "  No .sql files found in current directory"
    exit 1
fi

MIGRATION_FILE="$1"

# Check if file exists
if [ ! -f "$MIGRATION_FILE" ]; then
    echo -e "${RED}Error: Migration file '$MIGRATION_FILE' not found${NC}"
    exit 1
fi

echo -e "${YELLOW}=== Running Migration: $MIGRATION_FILE ===${NC}"
echo ""

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${RED}Error: Container '$CONTAINER_NAME' is not running${NC}"
    echo "Start it with: docker compose up -d postgres"
    exit 1
fi

# Run migration
echo -e "${GREEN}Executing migration on ${CONTAINER_NAME}...${NC}"
docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" < "$MIGRATION_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Migration completed successfully!${NC}"
else
    echo ""
    echo -e "${RED}❌ Migration failed!${NC}"
    exit 1
fi

# Show tables
echo ""
echo -e "${YELLOW}Current tables:${NC}"
docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -c "\dt weave_*"
