#!/bin/bash
set -euo pipefail

K="${1:?Usage: swap_reordered_data.sh <num_centroids>}"
REORDERED_BASE="/home/ec2-user/reordering-sweep-data"
SRC="${REORDERED_BASE}/kmeans_k${K}/data"
DEST="/home/ec2-user/before-reordering/after-reordering/data/nodes"
DISTRO="/home/ec2-user/k-NN-finn/build/testclusters/integTest-0/distro/3.6.0-ARCHIVE"

if [ ! -d "$SRC" ]; then
    echo "ERROR: ${SRC} does not exist. Available:"
    ls "${REORDERED_BASE}/"
    exit 1
fi

echo "Replacing after-reordering data with kmeans k=${K} ..."
rm -rf "$DEST"
cp -r "$SRC" "$DEST"

touch /tmp/dododo

echo "Prepping OpenSearch ..."
cd "$DISTRO"
rm -rf data
ln -s /home/ec2-user/before-reordering/data
cp ~/jvm.options ./config/jvm.options
cp ~/opensearch.yml ./config/opensearch.yml

echo "Ready. Start OpenSearch with: ${DISTRO}/bin/opensearch"
