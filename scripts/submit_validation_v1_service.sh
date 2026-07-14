#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SERVER="${SERVER:-http://127.0.0.1:9004}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Submit the bundled validation_v1/01 case to the standalone LTX MSR service,
wait for completion, and download the generated MP4.

Options:
  --output-dir DIR      Video output directory (default: ${PROJECT_ROOT}/outputs)
  --server URL          Service URL (default: http://127.0.0.1:9004)
  --python PATH         Python used to read JSON (default: /usr/bin/python3)
  --poll-interval SEC   Status polling interval (default: 5)
  -h, --help            Show this help

The same values can be set with OUTPUT_DIR, SERVER, PYTHON_BIN, and
POLL_INTERVAL environment variables.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --server)
            [[ $# -ge 2 ]] || { echo "--server requires a value" >&2; exit 2; }
            SERVER="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "--python requires a value" >&2; exit 2; }
            PYTHON_BIN="$2"
            shift 2
            ;;
        --poll-interval)
            [[ $# -ge 2 ]] || { echo "--poll-interval requires a value" >&2; exit 2; }
            POLL_INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

command -v curl >/dev/null || { echo "curl was not found" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || { echo "Python is not executable: ${PYTHON_BIN}" >&2; exit 1; }
SERVER="${SERVER%/}"

WORKFLOW="${PROJECT_ROOT}/sample_cases/LTX-2.3_MSR_sample_workflow_V2.json"
CASE_DIR="${PROJECT_ROOT}/sample_cases/validition_v1_01"

for required_file in \
    "${WORKFLOW}" \
    "${CASE_DIR}/1.jpg" \
    "${CASE_DIR}/2.jpg" \
    "${CASE_DIR}/bg.png"; do
    [[ -s "${required_file}" ]] || { echo "Required sample file is missing: ${required_file}" >&2; exit 1; }
done

GLOBAL_PROMPT=$("${PYTHON_BIN}" -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    workflow = json.load(stream)
node = next(item for item in workflow["nodes"] if item.get("type") == "PromptRelayEncode")
print(node["widgets_values"][0])
' "${WORKFLOW}")

LOCAL_PROMPTS=$("${PYTHON_BIN}" -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    workflow = json.load(stream)
node = next(item for item in workflow["nodes"] if item.get("type") == "PromptRelayEncode")
print(node["widgets_values"][1])
' "${WORKFLOW}")

echo "Checking service: ${SERVER}/health"
curl -fsS "${SERVER}/health"
echo

echo "Submitting validation_v1/01..."
SUBMIT_RESPONSE=$(curl -fsS -X POST "${SERVER}/submit" \
    -F 'pipeline_name=ltx_msr' \
    --form-string "global_prompt=${GLOBAL_PROMPT}" \
    --form-string "local_prompts=${LOCAL_PROMPTS}" \
    -F "subject_1=@${CASE_DIR}/2.jpg" \
    -F "subject_2=@${CASE_DIR}/1.jpg" \
    -F "background=@${CASE_DIR}/bg.png")

TASK_ID=$("${PYTHON_BIN}" -c '
import json, sys
response = json.loads(sys.argv[1])
task_id = response.get("task_id")
if not task_id:
    raise SystemExit(f"submit response has no task_id: {response}")
print(task_id)
' "${SUBMIT_RESPONSE}")

echo "Task submitted: ${TASK_ID}"
LAST_STATUS=""
while true; do
    STATUS_RESPONSE=$(curl -fsS "${SERVER}/status/${TASK_ID}")
    STATUS=$("${PYTHON_BIN}" -c '
import json, sys
response = json.loads(sys.argv[1])
print(response.get("status", "unknown"))
' "${STATUS_RESPONSE}")

    if [[ "${STATUS}" != "${LAST_STATUS}" ]]; then
        echo "Task status: ${STATUS}"
        LAST_STATUS="${STATUS}"
    fi

    case "${STATUS}" in
        done)
            break
            ;;
        error)
            echo "Task failed: ${STATUS_RESPONSE}" >&2
            exit 1
            ;;
        queued|running|submitted|unknown)
            sleep "${POLL_INTERVAL}"
            ;;
        *)
            echo "Unexpected task status: ${STATUS_RESPONSE}" >&2
            exit 1
            ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
OUTPUT_PATH="${OUTPUT_DIR}/validation_v1_01_${TASK_ID}.mp4"
TEMP_OUTPUT="${OUTPUT_PATH}.part"

curl -fL -o "${TEMP_OUTPUT}" "${SERVER}/download/${TASK_ID}"
mv "${TEMP_OUTPUT}" "${OUTPUT_PATH}"

echo "Video saved: ${OUTPUT_PATH}"
