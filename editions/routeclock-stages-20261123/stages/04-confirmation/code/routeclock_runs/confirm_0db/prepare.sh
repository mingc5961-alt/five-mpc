#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
assets=/data/chen/github_upload_refine_20261121/assets
stage() {
    local id=$1
    shift
    touch "state/$id.running"
    if "$@" >"logs/$id.log" 2>&1; then
        mv "state/$id.running" "state/$id.done"
    else
        mv "state/$id.running" "state/$id.failed"
        return 1
    fi
}
restore_code() {
    local checksum
    checksum=$(awk '$2=="rc-experiment-code.tar.zst"' "$assets/SHA256SUMS.txt")
    test -n "$checksum"
    (cd "$assets" && sha256sum -c <<< "$checksum")
    tar --zstd -xf "$assets/rc-experiment-code.tar.zst" -C work
}
fetch_amovfly_index() {
    local revision=67069ed00ddbebd62b71aa9bb1272415e9b15ff8
    curl -fL --retry 3 --max-time 900 "https://codeload.github.com/YujiaoHu/AMOVFLY-Dataset/tar.gz/$revision" -o work/amovfly.tar.gz.partial
    mkdir -p work/amovfly
    tar -xzf work/amovfly.tar.gz.partial --strip-components=1 -C work/amovfly
    test -s work/amovfly/Flight_info.csv
    mv work/amovfly.tar.gz.partial work/amovfly.tar.gz
    printf "%s\n" "$revision" > work/amovfly.revision
    find work/amovfly -maxdepth 2 -type f
}
date +%s > state/start_epoch
stage restore_code restore_code &
code_pid=$!
stage amovfly_index fetch_amovfly_index &
index_pid=$!
code_result=0
index_result=0
wait "$code_pid" || code_result=$?
wait "$index_pid" || index_result=$?
(( code_result == 0 && index_result == 0 ))
