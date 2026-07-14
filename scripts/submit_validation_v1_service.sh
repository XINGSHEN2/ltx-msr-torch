#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SERVER="${SERVER:-http://127.0.0.1:9004}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
GLOBAL_PROMPT_FILE="${GLOBAL_PROMPT_FILE:-}"
LOCAL_PROMPT_FILE="${LOCAL_PROMPT_FILE:-}"
NEGATIVE_PROMPT_FILE="${NEGATIVE_PROMPT_FILE:-}"
SUBJECT_1="${SUBJECT_1:-}"
SUBJECT_2="${SUBJECT_2:-}"
SUBJECT_3="${SUBJECT_3:-}"
SUBJECT_4="${SUBJECT_4:-}"
BACKGROUND="${BACKGROUND:-}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Submit an MSR case to the standalone LTX service, wait for completion, and
download the generated MP4. Prompt file options must be supplied together.

Options:
  --global-prompt-file PATH    UTF-8 global prompt text file
  --local-prompt-file PATH     UTF-8 local prompt text file
  --negative-prompt-file PATH  UTF-8 negative prompt text file
  --output-dir DIR             Video output directory (default: ${PROJECT_ROOT}/outputs)
  --background PATH            Required background image
  --subject-1 PATH             Optional first subject image
  --subject-2 PATH             Optional second subject image
  --subject-3 PATH             Optional third subject image
  --subject-4 PATH             Optional fourth subject image
  --server URL                 Service URL (default: http://127.0.0.1:9004)
  --python PATH                Python used to read JSON (default: /usr/bin/python3)
  --poll-interval SEC          Status polling interval (default: 5)
  -h, --help                   Show this help

Options can also be supplied with their uppercase environment variables, for
example GLOBAL_PROMPT_FILE, LOCAL_PROMPT_FILE, NEGATIVE_PROMPT_FILE, OUTPUT_DIR,
SUBJECT_1, BACKGROUND, SERVER, PYTHON_BIN, and POLL_INTERVAL.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --global-prompt-file)
            [[ $# -ge 2 ]] || { echo "--global-prompt-file requires a value" >&2; exit 2; }
            GLOBAL_PROMPT_FILE="$2"
            shift 2
            ;;
        --local-prompt-file)
            [[ $# -ge 2 ]] || { echo "--local-prompt-file requires a value" >&2; exit 2; }
            LOCAL_PROMPT_FILE="$2"
            shift 2
            ;;
        --negative-prompt-file)
            [[ $# -ge 2 ]] || { echo "--negative-prompt-file requires a value" >&2; exit 2; }
            NEGATIVE_PROMPT_FILE="$2"
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --subject-1)
            [[ $# -ge 2 ]] || { echo "--subject-1 requires a value" >&2; exit 2; }
            SUBJECT_1="$2"
            shift 2
            ;;
        --subject-2)
            [[ $# -ge 2 ]] || { echo "--subject-2 requires a value" >&2; exit 2; }
            SUBJECT_2="$2"
            shift 2
            ;;
        --subject-3)
            [[ $# -ge 2 ]] || { echo "--subject-3 requires a value" >&2; exit 2; }
            SUBJECT_3="$2"
            shift 2
            ;;
        --subject-4)
            [[ $# -ge 2 ]] || { echo "--subject-4 requires a value" >&2; exit 2; }
            SUBJECT_4="$2"
            shift 2
            ;;
        --background)
            [[ $# -ge 2 ]] || { echo "--background requires a value" >&2; exit 2; }
            BACKGROUND="$2"
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
[[ -n "${BACKGROUND}" ]] || { echo "--background is required" >&2; exit 2; }
[[ -s "${BACKGROUND}" ]] || { echo "Background image is missing or empty: ${BACKGROUND}" >&2; exit 1; }

for optional_image in "${SUBJECT_1}" "${SUBJECT_2}" "${SUBJECT_3}" "${SUBJECT_4}"; do
    if [[ -n "${optional_image}" && ! -s "${optional_image}" ]]; then
        echo "Optional subject image is missing or empty: ${optional_image}" >&2
        exit 1
    fi
done

PROMPT_FILE_COUNT=0
for prompt_file in "${GLOBAL_PROMPT_FILE}" "${LOCAL_PROMPT_FILE}" "${NEGATIVE_PROMPT_FILE}"; do
    if [[ -n "${prompt_file}" ]]; then
        PROMPT_FILE_COUNT=$((PROMPT_FILE_COUNT + 1))
    fi
done

if [[ "${PROMPT_FILE_COUNT}" -ne 0 && "${PROMPT_FILE_COUNT}" -ne 3 ]]; then
    echo "Provide all three prompt files together: global, local, and negative." >&2
    exit 2
fi

if [[ "${PROMPT_FILE_COUNT}" -eq 3 ]]; then
    for prompt_file in "${GLOBAL_PROMPT_FILE}" "${LOCAL_PROMPT_FILE}" "${NEGATIVE_PROMPT_FILE}"; do
        [[ -f "${prompt_file}" ]] || { echo "Prompt file was not found: ${prompt_file}" >&2; exit 1; }
    done
    GLOBAL_PROMPT="$(<"${GLOBAL_PROMPT_FILE}")"
    LOCAL_PROMPTS="$(<"${LOCAL_PROMPT_FILE}")"
    NEGATIVE_PROMPT="$(<"${NEGATIVE_PROMPT_FILE}")"
    [[ -n "${GLOBAL_PROMPT}" ]] || { echo "Global prompt file is empty" >&2; exit 1; }
    [[ -n "${LOCAL_PROMPTS}" ]] || { echo "Local prompt file is empty" >&2; exit 1; }
else
    [[ -s "${WORKFLOW}" ]] || { echo "Bundled workflow is missing: ${WORKFLOW}" >&2; exit 1; }
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

    NEGATIVE_PROMPT=$("${PYTHON_BIN}" -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    workflow = json.load(stream)
node = next(item for item in workflow["nodes"] if item.get("type") == "CLIPTextEncode")
print(node["widgets_values"][0])
' "${WORKFLOW}")
fi

echo "Checking service: ${SERVER}/health"
curl -fsS "${SERVER}/health"
echo

echo "Submitting LTX MSR task..."
CURL_ARGS=(
    -fsS -X POST "${SERVER}/submit"
    -F 'pipeline_name=ltx_msr'
    --form-string "global_prompt=${GLOBAL_PROMPT}"
    --form-string "local_prompts=${LOCAL_PROMPTS}"
    --form-string "negative_prompt=${NEGATIVE_PROMPT}"
    --form-string "output_dir=${OUTPUT_DIR}"
    -F "background=@${BACKGROUND}"
)
for subject_number in 1 2 3 4; do
    subject_variable="SUBJECT_${subject_number}"
    subject_path="${!subject_variable}"
    if [[ -n "${subject_path}" ]]; then
        CURL_ARGS+=(-F "subject_${subject_number}=@${subject_path}")
    fi
done
SUBMIT_RESPONSE=$(curl "${CURL_ARGS[@]}")

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
OUTPUT_PATH="${OUTPUT_DIR}/ltx_msr_${TASK_ID}.mp4"
TEMP_OUTPUT="${OUTPUT_PATH}.part"

curl -fL -o "${TEMP_OUTPUT}" "${SERVER}/download/${TASK_ID}"
mv "${TEMP_OUTPUT}" "${OUTPUT_PATH}"

echo "Video saved: ${OUTPUT_PATH}"
