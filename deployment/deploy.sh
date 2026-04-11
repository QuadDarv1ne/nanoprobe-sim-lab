#!/bin/bash
# Nanoprobe Sim Lab - Production Deployment Script
#
# Usage:
#   ./deploy.sh [up|down|restart|logs|status]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="deployment/docker-compose.prod.yml"
PROJECT_NAME="nanoprobe"
ENV_FILE="deployment/.env"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_env() {
    if [ ! -f "$ENV_FILE" ]; then
        log_warn ".env file not found. Copying from .env.example..."
        cp "$ENV_FILE.example" "$ENV_FILE"
        log_warn "Please edit $ENV_FILE with your production values!"
        exit 1
    fi
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
}

deploy() {
    log_info "Starting deployment..."

    # Check prerequisites
    check_docker
    check_env

    # Pull latest images
    log_info "Pulling latest images..."
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME pull

    # Run migrations
    log_info "Running database migrations..."
    # docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME run --rm migration

    # Start services
    log_info "Starting services..."
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d

    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 10

    # Check health
    log_info "Checking service health..."
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME ps

    log_info "Deployment completed successfully!"
}

stop() {
    log_info "Stopping all services..."
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down
}

restart() {
    stop
    sleep 5
    deploy
}

logs() {
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f
}

status() {
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME ps
}

# Main
case "${1:-up}" in
    up)
        deploy
        ;;
    down)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        logs
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {up|down|restart|logs|status}"
        exit 1
        ;;
esac
