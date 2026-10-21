#!/bin/bash
# Source from the native recovery launcher. Stream one pair at a time so dense
# raw assets never accumulate across the test set. Sampling seeds and camera
# settings stay in the frozen workers; sharding only selects pair identities.

dense_process_method() {
    local cell="$1" method="$2" cfg="$3" slat_cfg="$4" phase="$5"
    local pair_id pair_dir pair_list="$cell/dense_pairs.tsv" pair_phase
    if [[ -s "$cell/$method/raw_retired.json" ]]; then
        run_phase "${phase}_verify_retained" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" \
            "$MORPHFLOW_DENSE_RETENTION_SCRIPT" --run-root "$cell" --method "$method" --check
        run_phase "${phase}_finish_retirement" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" \
            "$MORPHFLOW_DENSE_RETENTION_SCRIPT" --run-root "$cell" --method "$method" --release
        return
    fi
    run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); ids=[int(r["pair_id"]) for r in p["pairs"]]; assert sorted(ids)==list(range(len(ids))); print("\n".join(str(r["pair_id"])+"\t"+r["pair_dir"] for r in p["pairs"]))' \
        "$cell/plan.json" > "$pair_list"
    while IFS=$'\t' read -r pair_id pair_dir; do
        pair_phase="${phase}_pair_${pair_id}"
        if [[ -s "$cell/$method/$pair_dir/raw_retired.json" ]]; then
            run_phase "${pair_phase}_verify_retained" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" \
                "$MORPHFLOW_DENSE_RETENTION_SCRIPT" --run-root "$cell" --method "$method" --pair "$pair_dir" --check
            run_phase "${pair_phase}_finish_retirement" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" \
                "$MORPHFLOW_DENSE_RETENTION_SCRIPT" --run-root "$cell" --method "$method" --pair "$pair_dir" --release
            continue
        fi
        case "$method" in
            morphflow)
                run_phase "${pair_phase}_generation" "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$SCRIPT" morphflow \
                    --run-root "$cell" --data-root "$DATA_ROOT_HOST" --metadata "$RUN_ROOT/input_snapshot/metadata_test.json" \
                    --source-images-root "$SOURCE_IMAGES_ROOT_HOST" --source-image-filename "$SOURCE_IMAGE_FILENAME" \
                    --checkpoint-path "$RUN_ROOT/checkpoints/ss/checkpoint.pt" --slat-checkpoint-path "$RUN_ROOT/checkpoints/slat/checkpoint.pt" \
                    --steps "$STEPS" --slat-steps "$SLAT_STEPS" --cfg-scale "$cfg" --slat-cfg-scale "$slat_cfg" \
                    --mixed-precision "$MIXED_PRECISION" --trellis-model image_large \
                    --shard-index "$pair_id" --num-shards "$ACTUAL_PAIRS"
                ;;
            morphany3d)
                run_phase "${pair_phase}_generation" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$SCRIPT" morphany3d \
                    --run-root "$cell" --steps "$STEPS" --slat-steps "$SLAT_STEPS" --tfsa-cache-mode hybrid \
                    --tfsa-ram-gb "$MORPHANY_TFSA_RAM_GB" --tfsa-min-node-free-gb "$MORPHANY_TFSA_MIN_FREE_GB" \
                    --local-cache-root "$TASK_TMP/morphany3d" --tfsa-spill-root "$TASK_TMP/tfsa-spill" \
                    --shard-index "$pair_id" --num-shards "$ACTUAL_PAIRS"
                ;;
            interp3d)
                run_phase "${pair_phase}_generation" "$INTERP_ENV" "$SNAPSHOT/Interp3D" "$SCRIPT" interp3d \
                    --run-root "$cell" --steps "$STEPS" --slat-steps "$SLAT_STEPS" \
                    --shard-index "$pair_id" --num-shards "$ACTUAL_PAIRS"
                ;;
            *) die "Unknown dense method: $method" ;;
        esac
        run_phase "${pair_phase}_metrics" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$EVALUATOR" shard \
            --run-dir "$cell/$method" --output-dir "$cell/metrics/$method" --stage all \
            --shard-index "$pair_id" --num-shards "$ACTUAL_PAIRS" \
            --num-views "$NUM_VIEWS" --resolution 512 --camera-distribution hammersley --render-backend gaussian \
            --appearance materials --lpips-net vgg --lpips-batch-size "$LPIPS_BATCH_SIZE" \
            --metric-mode weighted-endpoints-only --reference-cache-root "$RUN_ROOT/shared/reference_metrics" --device cuda
        run_phase "${pair_phase}_retire_raw" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" \
            "$MORPHFLOW_DENSE_RETENTION_SCRIPT" --run-root "$cell" --method "$method" --pair "$pair_dir" --release
    done < "$pair_list"
    run_phase "${phase}_merge" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$EVALUATOR" merge \
        --output-dir "$cell/metrics/$method" --metric-mode weighted-endpoints-only --expected-num-pairs "$ACTUAL_PAIRS"
    run_phase "${phase}_retire_raw" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" \
        "$MORPHFLOW_DENSE_RETENTION_SCRIPT" --run-root "$cell" --method "$method" --release
}

run_dense_streaming_grid() {
    local base="$RUN_ROOT/weighted_50" cfg slat_cfg cell phase method
    # Canonical references are shared with the already completed seven-state run.
    for method in morphany3d interp3d; do
        dense_process_method "$base/shared" "$method" "" "" "weighted50_${method}"
    done
    while read -r cfg slat_cfg; do
        cell="$(run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$HELPER" cell --run-root "$RUN_ROOT" --evaluation weighted50 --cfg "$cfg" --slat-cfg "$slat_cfg")"
        phase="weighted50_$(basename "$cell")"
        if [[ -s "$cell/final_metrics.json" && -s "$cell/final_metrics.csv" ]]; then
            echo "RESUME: $phase already completed; retained renders and metrics"
            continue
        fi
        dense_process_method "$cell" morphflow "$cfg" "$slat_cfg" "$phase"
        run_phase "${phase}_aggregate" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$SCRIPT" aggregate --run-root "$cell"
        run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$HELPER" summarize --run-root "$RUN_ROOT" --evaluation weighted50
        run_phase "${phase}_weighted_report" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" \
            "$ACTIVE_PROJECT/benchmarks/weighted_endpoint_report.py" --run-root "$base"
    done < "$GRID_FILE"
    run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$HELPER" summarize --run-root "$RUN_ROOT" --evaluation weighted50 --require-complete
}
