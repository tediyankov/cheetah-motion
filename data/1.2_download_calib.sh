#!/bin/bash

set -euo pipefail

# config -------------------------------------------------------------------------

DEST_DIR="/gws/nopw/j04/iecdt/cheetah"
TMP_DIR="${DEST_DIR}/.tmp_dropbox_dl"
MAX_RETRIES=3

# entries ------------------------------------------------------------------------

declare -a ENTRIES=(
    "https://www.dropbox.com/scl/fo/s5hvgihoqvkrieu8i9mtn/AEXeJzn1jOQ5HXwxMLFiqDU/2017_08_29?rlkey=g19j1jwqv3cgsdvt9kh7nti6y&st=itkb9kof&dl=1|||2017_08_29"
    "https://www.dropbox.com/scl/fo/s5hvgihoqvkrieu8i9mtn/AEkYh0-WrLItvWbaAekfQA0/2017_09_02?rlkey=g19j1jwqv3cgsdvt9kh7nti6y&subfolder_nav_tracking=1&st=y5fqpbz1&dl=1|||2017_09_02"
    "https://www.dropbox.com/scl/fo/s5hvgihoqvkrieu8i9mtn/AN5-LIbbIsEh2hTR1i36m6E/2017_09_03?rlkey=g19j1jwqv3cgsdvt9kh7nti6y&subfolder_nav_tracking=1&st=h70y45g8&dl=0|||2017_09_03"
    "https://www.dropbox.com/scl/fo/s5hvgihoqvkrieu8i9mtn/AL82yAbI594AwMJleVZdJ8Q/2017_12_09?rlkey=g19j1jwqv3cgsdvt9kh7nti6y&subfolder_nav_tracking=1&st=z6z13tro&dl=0|||2017_12_09"
    "https://www.dropbox.com/scl/fo/s5hvgihoqvkrieu8i9mtn/AHLLLQJny5dZJ6PifLhbMsE/2017_12_10?rlkey=g19j1jwqv3cgsdvt9kh7nti6y&subfolder_nav_tracking=1&st=q5q7lb46&dl=0|||2017_12_10"
    "https://www.dropbox.com/scl/fo/s5hvgihoqvkrieu8i9mtn/AIgafMjM2oh6VAWLqEoxJ_4/2017_12_12?rlkey=g19j1jwqv3cgsdvt9kh7nti6y&subfolder_nav_tracking=1&st=kufa3axq&dl=0|||2017_12_12"
    "https://www.dropbox.com/scl/fo/s5hvgihoqvkrieu8i9mtn/AINLy_yah9_YJGKxE9IoYb0/2017_12_14?rlkey=g19j1jwqv3cgsdvt9kh7nti6y&subfolder_nav_tracking=1&st=cqp2hbai&dl=0|||2017_12_14"
    "https://www.dropbox.com/scl/fo/s5hvgihoqvkrieu8i9mtn/AHInnQz9kat0ok8L6y9nNeY/2017_12_16?rlkey=g19j1jwqv3cgsdvt9kh7nti6y&subfolder_nav_tracking=1&st=7csiummn&dl=0|||2017_12_16"
    "https://www.dropbox.com/scl/fo/s5hvgihoqvkrieu8i9mtn/ANWGegJWlcxvIufYbExMqlI/2017_12_17?rlkey=g19j1jwqv3cgsdvt9kh7nti6y&subfolder_nav_tracking=1&st=ape84o1x&dl=0|||2017_12_17"
    "https://www.dropbox.com/scl/fo/s5hvgihoqvkrieu8i9mtn/ACjNpqzoI9Yueu_peTrKoDI/2017_12_21?rlkey=g19j1jwqv3cgsdvt9kh7nti6y&subfolder_nav_tracking=1&st=ip4cuxwy&dl=0|||2017_12_21"
    "https://www.dropbox.com/scl/fo/s5hvgihoqvkrieu8i9mtn/AFZ-Oi3Sn1L_cA-jdCWXR4Y/2019_02_27?rlkey=g19j1jwqv3cgsdvt9kh7nti6y&subfolder_nav_tracking=1&st=a4y6e32t&dl=0|||2019_02_27"
    "https://www.dropbox.com/scl/fo/s5hvgihoqvkrieu8i9mtn/AOmN-C8TfY564oijaUH9ZR8/2019_03_03?rlkey=g19j1jwqv3cgsdvt9kh7nti6y&subfolder_nav_tracking=1&st=fvwiym6t&dl=0|||2019_03_03"
    "https://www.dropbox.com/scl/fo/s5hvgihoqvkrieu8i9mtn/AN--qR2g47AwzWWFvhckhM0/2019_03_05?rlkey=g19j1jwqv3cgsdvt9kh7nti6y&subfolder_nav_tracking=1&st=ih861z8q&dl=0|||2019_03_05"
    "https://www.dropbox.com/scl/fo/s5hvgihoqvkrieu8i9mtn/AOVZPqL4QdBdhz7L8GcQ_pI/2019_03_07?rlkey=g19j1jwqv3cgsdvt9kh7nti6y&subfolder_nav_tracking=1&st=y9mkpttb&dl=0|||2019_03_07"
    "https://www.dropbox.com/scl/fo/s5hvgihoqvkrieu8i9mtn/AApyWU6RmSa7XyF_mf76lo4/2019_03_09?rlkey=g19j1jwqv3cgsdvt9kh7nti6y&subfolder_nav_tracking=1&st=sriewbri&dl=0|||2019_03_09"
)

# helpers ------------------------------------------------------------------------

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()   { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn()  { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*" >&2; }

## main --------------------------------------------------------------------------

if [[ ${#ENTRIES[@]} -eq 0 ]]; then
    error "No entries defined in ENTRIES array. Please fill in the script."
    exit 1
fi

mkdir -p "$DEST_DIR"
mkdir -p "$TMP_DIR"

FAILED=()
SUCCESS=0

for entry in "${ENTRIES[@]}"; do
    URL="${entry%%|||*}"
    DATE_FOLDER="${entry##*|||}"

    ZIP_FILE="${TMP_DIR}/${DATE_FOLDER}.zip"
    EXTRACT_DIR="${TMP_DIR}/${DATE_FOLDER}_extracted"
    FINAL_DIR="${DEST_DIR}/${DATE_FOLDER}"

    log "==============================================================="
    log "Processing: ${DATE_FOLDER}"

    # 1. download
    ATTEMPT=0
    DOWNLOAD_OK=false
    while [[ $ATTEMPT -lt $MAX_RETRIES ]]; do
        ((ATTEMPT++)) || true
        log "  Downloading (attempt ${ATTEMPT}/${MAX_RETRIES})..."
        if curl -fsSL \
                --retry 3 --retry-delay 5 \
                --max-time 7200 \
                -o "$ZIP_FILE" \
                "$URL"; then
            DOWNLOAD_OK=true
            break
        else
            warn "  Attempt ${ATTEMPT} failed. Waiting 15s..."
            sleep 15
        fi
    done

    if [[ "$DOWNLOAD_OK" != true ]]; then
        error "  Download failed after ${MAX_RETRIES} attempts. Skipping ${DATE_FOLDER}."
        FAILED+=("$DATE_FOLDER")
        rm -f "$ZIP_FILE"
        continue
    fi

    # sanity check 
    if ! file "$ZIP_FILE" | grep -q "Zip archive"; then
        error "  Downloaded file is not a zip (Dropbox may have returned an error page)."
        error "  Check URL: ${URL}"
        FAILED+=("$DATE_FOLDER")
        rm -f "$ZIP_FILE"
        continue
    fi

    # 2. unzip
    log "  Unzipping..."
    rm -rf "$EXTRACT_DIR"
    mkdir -p "$EXTRACT_DIR"

    UNZIP_OK=false
    
    # trying standard unzip
    export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
    if unzip -q -o "$ZIP_FILE" -d "$EXTRACT_DIR" 2>/dev/null; then
        UNZIP_OK=true
    else
        warn "  Standard unzip failed, trying Python zipfile..."
        # if that doesnt work try python zipfile
        if python3 << PYPYTHON
import zipfile
import sys
try:
    with zipfile.ZipFile("$ZIP_FILE", 'r') as zip_ref:
        zip_ref.extractall("$EXTRACT_DIR")
    sys.exit(0)
except Exception as e:
    print(f"Python unzip failed: {e}", file=sys.stderr)
    sys.exit(1)
PYPYTHON
        then
            UNZIP_OK=true
            log "  Python unzip succeeded."
        fi
    fi
    unset UNZIP_DISABLE_ZIPBOMB_DETECTION

    if [[ "$UNZIP_OK" != true ]]; then
        error "  Both unzip methods failed for ${DATE_FOLDER}."
        FAILED+=("$DATE_FOLDER")
        rm -f "$ZIP_FILE"
        rm -rf "$EXTRACT_DIR"
        continue
    fi

    # removing zip immediately to save space
    rm -f "$ZIP_FILE"
    log "  Zip deleted. Scanning contents..."

    # 3. finding all extrinsic_calib directories anywhere in the tree
    N_TOP=$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
    if [[ $N_TOP -eq 1 ]]; then
        SEARCH_ROOT=$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | head -1)
        log "  Stepping into single top-level folder: $(basename "$SEARCH_ROOT")"
    else
        SEARCH_ROOT="$EXTRACT_DIR"
    fi

    TARGETS_FOUND=0

    while IFS= read -r -d '' target_dir; do
        # relative path e.g. bottom/extrinsic_calib
        REL_PATH="${target_dir#"${SEARCH_ROOT}/"}"

        # dest path
        DEST_PATH="${FINAL_DIR}/${REL_PATH}"

        log "    Copying: ${REL_PATH}"
        mkdir -p "$(dirname "$DEST_PATH")"

        if rsync -a "$target_dir/" "$DEST_PATH/" 2>/dev/null; then
            ((TARGETS_FOUND++)) || true
        else
            warn "    rsync failed for ${REL_PATH}, falling back to cp..."
            cp -r "$target_dir" "$DEST_PATH"
            ((TARGETS_FOUND++)) || true
        fi

        # remove heavy subfolders inside destination extrinsic_calib
        rm -rf "$DEST_PATH/frames" "$DEST_PATH/points" "$DEST_PATH/videos"

    done < <(find "$SEARCH_ROOT" -type d -name "extrinsic_calib" -print0)

    if [[ $TARGETS_FOUND -eq 0 ]]; then
        warn "  No extrinsic_calib folders found inside ${DATE_FOLDER}. Check the zip contents."
    else
        log "  Copied ${TARGETS_FOUND} extrinsic_calib folder(s)."
    fi

    # 4. cleaning up extracted folder
    rm -rf "$EXTRACT_DIR"
    log "  Cleaned up. Done with ${DATE_FOLDER}."

    ((SUCCESS++)) || true
done

# final cleanup of temp dir if empty
rmdir "$TMP_DIR" 2>/dev/null || true

log "==============================================================="
log "Finished: ${SUCCESS} date folder(s) succeeded, ${#FAILED[@]} failed."
if [[ ${#FAILED[@]} -gt 0 ]]; then
    error "Failed folders:"
    for f in "${FAILED[@]}"; do error "  - $f"; done
    exit 1
fi