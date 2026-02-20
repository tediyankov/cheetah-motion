!/bin/bash

eval "$(mamba shell hook --shell bash)"
mamba activate ~/envs/cheetah-motion-env

# setting directories
BASE_DATA_DIR="/gws/nopw/j04/iecdt/cheetah"
UNDISTORT_SCRIPT="/gws/nopw/j04/iecdt/tyankov/cheetah-motion/data/2_calibration.py"
FILTER_SCRIPT="/gws/nopw/j04/iecdt/tyankov/cheetah-motion/data/3_filtering2D.py"

# config
CONF_THRESHOLD=0.3

# counters
TOTAL_PROCESSED=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0
FAILED_LIST=()

echo "=========================================="
echo "Starting preprocessing pipeline"
echo "Base directory: ${BASE_DATA_DIR}"
echo "=========================================="
echo ""

# if you wanna put ur own sequences in here then do it, otherise remember to leave this EMPTY
MANUAL_SEQUENCES=(
    # "2017_09_03/top/phantom/run1"
    # "2017_12_17/top/zorro/flick1"
)

if [ ${#MANUAL_SEQUENCES[@]} -gt 0 ]; then
    SEQ_LIST=$(printf "%s\n" "${MANUAL_SEQUENCES[@]}")
else
    SEQ_LIST=$(python3 - <<'PY'
from pathlib import Path
import re

def is_date_folder(path: Path) -> bool:
    return path.is_dir() and re.fullmatch(r"\d{4}_\d{2}_\d{2}", path.name)

def find_all_sequences(dataset_root: str):
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"{dataset_root} does not exist")
    sequences = set()
    for date_dir in sorted(root.iterdir()):
        if not is_date_folder(date_dir):
            continue
        for fte_dir in date_dir.rglob("fte_pw"):
            seq_dir = fte_dir.parent
            rel_path = seq_dir.relative_to(root)
            sequences.add(rel_path.as_posix())
    return sorted(sequences)

dataset_root = "/gws/nopw/j04/iecdt/cheetah"
for s in find_all_sequences(dataset_root):
    print(s)
PY
)
fi

if [[ -z "${SEQ_LIST}" ]]; then
    echo "No sequences found (fte_pw not found). Exiting."
    exit 1
fi

while IFS= read -r REL_PATH; do
    [[ -z "${REL_PATH}" ]] && continue
    seq_dir="${BASE_DATA_DIR}/${REL_PATH}"

    # DLC folder
    DLC_DIR="${seq_dir}/dlc"
    if [ ! -d "${DLC_DIR}" ]; then
        continue
    fi

    # find nearest extrinsic_calib by walking up
    EXTRINSIC_DIR=""
    SEARCH_DIR="${seq_dir}"
    while [[ "${SEARCH_DIR}" != "${BASE_DATA_DIR}" && "${SEARCH_DIR}" != "/" ]]; do
        if [[ -d "${SEARCH_DIR}/extrinsic_calib" ]]; then
            EXTRINSIC_DIR="${SEARCH_DIR}/extrinsic_calib"
            break
        fi
        SEARCH_DIR="$(dirname "${SEARCH_DIR}")"
    done

    if [ -z "${EXTRINSIC_DIR}" ]; then
        echo "     WARNING: No extrinsic_calib found for ${REL_PATH}"
        continue
    fi

    # counting number of unique camera h5 files
    N_CAMS=$(ls "${DLC_DIR}"/cam*.h5 2>/dev/null | wc -l)

    if [ "${N_CAMS}" -eq 0 ]; then
        echo "     WARNING: No cam*.h5 files found in ${DLC_DIR}"
        ((TOTAL_SKIPPED++))
        continue
    fi

    # determining which scene JSON to use based on number of cameras
    SCENE_JSON="${EXTRINSIC_DIR}/${N_CAMS}_cam_scene_sba.json"

    if [ ! -f "${SCENE_JSON}" ]; then
        # fallback: if only one JSON exists, use it
        JSON_COUNT=$(ls "${EXTRINSIC_DIR}"/*_cam_scene_sba.json 2>/dev/null | wc -l)
        if [ "${JSON_COUNT}" -eq 1 ]; then
            SCENE_JSON=$(ls "${EXTRINSIC_DIR}"/*_cam_scene_sba.json)
            echo "     WARNING: No ${N_CAMS}_cam_scene_sba.json found. Using ${SCENE_JSON}"
        else
            echo "     ERROR: Scene JSON not found: ${SCENE_JSON}"
            echo "            (Found ${N_CAMS} cameras but no matching JSON)"
            ((TOTAL_FAILED++))
            FAILED_LIST+=("${REL_PATH} [missing scene json: ${SCENE_JSON}]")
            continue
        fi
    fi

    echo ""
    echo "     ----------------------------------------"
    echo "     Processing: ${REL_PATH}"
    echo "     Cameras: ${N_CAMS}"
    echo "     Scene JSON: ${N_CAMS}_cam_scene_sba.json"
    echo "     ----------------------------------------"

    # step 1: undistorting
    echo "     [1/2] Undistorting keypoints..."

    python "${UNDISTORT_SCRIPT}" \
        --dlc-dir "${DLC_DIR}" \
        --scene-json "${SCENE_JSON}" 2>&1 | sed 's/^/       /'

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "     ERROR: Undistortion failed"
        ((TOTAL_FAILED++))
        FAILED_LIST+=("${REL_PATH} [undistort failed]")
        continue
    fi

    # step 2: filtering by confidence
    UNDISTORTED_DIR="${seq_dir}/undistorted_2D"

    echo "     [2/2] Filtering by confidence (threshold=${CONF_THRESHOLD})..."

    python "${FILTER_SCRIPT}" \
        --undistorted-dir "${UNDISTORTED_DIR}" \
        --conf-threshold "${CONF_THRESHOLD}" 2>&1 | sed 's/^/       /'

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "     ERROR: Filtering failed"
        ((TOTAL_FAILED++))
        FAILED_LIST+=("${REL_PATH} [filtering failed]")
        continue
    fi

    echo "     ✓ Successfully processed"
    ((TOTAL_PROCESSED++))
    echo ""

done <<< "${SEQ_LIST}"

echo "=========================================="
echo "Pipeline complete"
echo "=========================================="
echo "Successfully processed: ${TOTAL_PROCESSED}"
echo "Failed: ${TOTAL_FAILED}"
echo "Skipped (no DLC data): ${TOTAL_SKIPPED}"
if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo "Failed items:"
    for item in "${FAILED_LIST[@]}"; do
        echo "  - ${item}"
    done
fi
echo "=========================================="