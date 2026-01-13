#!/bin/bash

set -e

MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
TRAINING_SCRIPT=${TRAINING_SCRIPT:-train.py}
HEADLESS_SERVICE="${NF_DISCOVERY_SERVICE}"

log() {
    echo "[$(date -u +%H:%M:%S)] $1"
}

if [ -z "${HEADLESS_SERVICE}" ]; then
    log "ERROR: NF_DISCOVERY_SERVICE not set"
    exit 1
fi

if [ -z "${REPLICAS}" ]; then
    log "ERROR: REPLICAS not set"
    exit 1
fi

HOSTNAME=$(hostname)
NODE_RANK=${HOSTNAME##*-}
BASE_NAME=${HOSTNAME%-*}
MASTER_HOSTNAME="${BASE_NAME}-0"
MASTER_ADDR="${MASTER_HOSTNAME}.${HEADLESS_SERVICE}"

log "Node ${NODE_RANK}/${REPLICAS} starting, master=${MASTER_ADDR}"
log "Starting torchrun (rendezvous will synchronize nodes)"

exec torchrun \
    --nnodes=${REPLICAS} \
    --nproc-per-node=${NPROC_PER_NODE} \
    --node-rank=${NODE_RANK} \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    ${TRAINING_SCRIPT} ${SCRIPT_ARGS}
