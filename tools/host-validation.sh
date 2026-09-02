#!/usr/bin/env bash
# Pull-based host validation (M1144): a host that owns hardware CI cannot reach
# (an RTX 3060 desktop, a headless bench box) fast-forwards its own checkout to
# master on its own schedule, runs one suite, and posts a commit status on the
# SHA it tested. Control is inverted on purpose: there is no runner daemon and
# nothing on GitHub can start a process here. The only trigger is the systemd
# user timer in tools/systemd, and it only ever runs code already on master.
#
# Usage: tools/host-validation.sh <desktop-gpu|bench> [--dry-run]
#   --dry-run prints the status payload instead of posting it, and needs no token.
#
# The token is a fine-grained PAT scoped to this repository with only the
# "Commit statuses" permission (read and write), read from the file named by
# $G2G_VALIDATION_TOKEN_FILE (default ~/.config/glass2glass-validation/token,
# which must be mode 0600 and owned by the caller). It is never echoed, never
# placed on a command line, and only ever sent to api.github.com.
#
# Exit 0 if nothing FAILed (SKIPs are allowed), 1 otherwise.
set -uo pipefail

# xtrace would echo the token into the journal.
set +x

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

GIT_REMOTE="origin"
VALIDATED_BRANCH="master"
GITHUB_API_BASE="https://api.github.com"
GITHUB_API_VERSION="2022-11-28"
HTTP_CREATED="201"
STATUS_CONTEXT_PREFIX="host-validation"
STATUS_DESCRIPTION_MAX_CHARACTERS=140
STATUS_POST_TIMEOUT_SECONDS=30
DEFAULT_TOKEN_FILE="$HOME/.config/glass2glass-validation/token"
REQUIRED_TOKEN_FILE_MODE="600"
FAILURE_LOG_TAIL_LINES=25

# wgpu-sink and hdr-present are what the three vulkan test files that need more
# than vulkan-video gate on. Without them those files compile to empty binaries.
VULKAN_VIDEO_FEATURES="vulkan-video,wgpu-sink,hdr-present"
CUDA_FEATURES="nvdec,nvenc,cuda-wgpu,ffmpeg"
CUDA_WGPU_END_TO_END_FEATURES="cuda-wgpu-e2e"
SOAK_FEATURES="hls ffmpeg wayland-sink pipewire"
# The CUDA/NVDEC test files share no filename pattern, unlike the vulkan ones.
CUDA_TEST_TARGETS=(
  m352_nvdec_domain_negotiation
  m353_cuda_upload
  m1062_cuda_device_id
  cudawgpu_spike
  wgpu_to_cuda
)

# cuda_wgpu_e2e skips itself when G2G_H264_FIXTURE is unset, so the suite
# generates the clip its header describes and caches it.
FIXTURE_DIRECTORY="${FIXTURE_DIRECTORY:-/tmp/g2g-host-validation-fixtures}"
FIXTURE_WIDTH=320
FIXTURE_HEIGHT=240
FIXTURE_FRAMES_PER_SECOND=30
FIXTURE_DURATION_SECONDS=1
FIXTURE_KEYFRAME_INTERVAL=15
H264_FIXTURE="$FIXTURE_DIRECTORY/testsrc-${FIXTURE_WIDTH}x${FIXTURE_HEIGHT}p${FIXTURE_FRAMES_PER_SECOND}-${FIXTURE_DURATION_SECONDS}s.h264"

# The soak reads its feed from the same variable the test does.
HLS_TEST_URL="${G2G_HLS_TEST_URL:-http://localhost:8888/avpattern/index.m3u8}"
HLS_PROBE_TIMEOUT_SECONDS=5

# Overridable the way qualification-kit.sh overrides qemu-system-arm.
GSTREAMER_LAUNCH="${G2G_VALIDATION_GST_LAUNCH:-gst-launch-1.0}"

usage() {
  echo "usage: tools/host-validation.sh <desktop-gpu|bench> [--dry-run]" >&2
}

SUITE_NAME=""
DRY_RUN=0
for argument in "$@"; do
  case "$argument" in
    --dry-run) DRY_RUN=1 ;;
    desktop-gpu|bench) SUITE_NAME="$argument" ;;
    *) usage; exit 2 ;;
  esac
done
if [ -z "$SUITE_NAME" ]; then
  usage
  exit 2
fi
STATUS_CONTEXT="$STATUS_CONTEXT_PREFIX/$SUITE_NAME"

WORK_DIRECTORY="$(mktemp -d)"
trap 'rm -rf "$WORK_DIRECTORY"' EXIT

# ---------------------------------------------------------------- token

GITHUB_TOKEN=""

read_github_token() {
  local token_file="${G2G_VALIDATION_TOKEN_FILE:-$DEFAULT_TOKEN_FILE}"
  if [ ! -f "$token_file" ]; then
    echo "token file not found: $token_file" >&2
    return 1
  fi
  if [ ! -O "$token_file" ]; then
    echo "token file $token_file is not owned by $(id -un)" >&2
    return 1
  fi
  local file_mode
  file_mode="$(stat -c '%a' "$token_file")"
  if [ "$file_mode" != "$REQUIRED_TOKEN_FILE_MODE" ]; then
    echo "token file $token_file is mode $file_mode, want $REQUIRED_TOKEN_FILE_MODE" >&2
    return 1
  fi
  IFS= read -r GITHUB_TOKEN <"$token_file"
  if [ -z "$GITHUB_TOKEN" ]; then
    echo "token file $token_file is empty" >&2
    return 1
  fi
}

# ---------------------------------------------------------------- checkout

github_repository_from_remote() {
  local remote_url slug
  remote_url="$(git -C "$REPOSITORY_ROOT" remote get-url "$GIT_REMOTE")" || return 1
  slug="$remote_url"
  slug="${slug#https://github.com/}"
  slug="${slug#git@github.com:}"
  slug="${slug%.git}"
  # A remote that is not on github.com leaves the URL unchanged, and the token
  # must never be offered for a repository this endpoint does not serve.
  if [ "$slug" = "$remote_url" ] || ! [[ "$slug" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
    echo "$GIT_REMOTE '$remote_url' is not a github.com repository" >&2
    return 1
  fi
  printf '%s' "$slug"
}

# A change to this script takes effect on the run after it lands: this run
# already read the previous version.
update_checkout_to_validated_branch() {
  local current_branch
  current_branch="$(git -C "$REPOSITORY_ROOT" symbolic-ref --quiet --short HEAD)" || current_branch=""
  if [ "$current_branch" != "$VALIDATED_BRANCH" ]; then
    echo "checkout is on '${current_branch:-a detached HEAD}', want $VALIDATED_BRANCH" >&2
    return 1
  fi
  if [ -n "$(git -C "$REPOSITORY_ROOT" status --porcelain)" ]; then
    echo "checkout has local modifications: a status would name a SHA that is not what ran" >&2
    return 1
  fi
  # Without a terminal git would otherwise sit waiting for credentials, and a
  # timer run has nobody to type them.
  GIT_TERMINAL_PROMPT=0 git -C "$REPOSITORY_ROOT" fetch --quiet "$GIT_REMOTE" "$VALIDATED_BRANCH" || return 1
  git -C "$REPOSITORY_ROOT" merge --quiet --ff-only "$GIT_REMOTE/$VALIDATED_BRANCH" || return 1
}

# ---------------------------------------------------------------- steps

STEP_NAMES=()
STEP_RESULTS=()
STEP_DETAILS=()
PASSED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0
SKIP_REASON=""
LAST_STEP_LOG=""
LAST_STEP_RESULT=""

record_step() {
  STEP_NAMES+=("$1")
  STEP_RESULTS+=("$2")
  STEP_DETAILS+=("$3")
  LAST_STEP_RESULT="$2"
  case "$2" in
    PASS) PASSED_COUNT=$((PASSED_COUNT + 1)) ;;
    SKIP) SKIPPED_COUNT=$((SKIPPED_COUNT + 1)) ;;
    FAIL) FAILED_COUNT=$((FAILED_COUNT + 1)) ;;
  esac
}

set_last_step_detail() {
  STEP_DETAILS[$((${#STEP_DETAILS[@]} - 1))]="$1"
}

# Records the SKIP and returns 1 when the step must not run, 0 otherwise.
# A precondition function returns non-zero and sets SKIP_REASON to skip.
step_precondition_met() {
  local step_name="$1" precondition="$2"
  if [ "$precondition" = "always" ]; then
    return 0
  fi
  SKIP_REASON=""
  if "$precondition"; then
    return 0
  fi
  record_step "$step_name" "SKIP" "$SKIP_REASON"
  echo "== $step_name: SKIP ($SKIP_REASON) =="
  return 1
}

# run_step <name> <precondition-function|always> <command...>
run_step() {
  local step_name="$1" precondition="$2"
  shift 2
  step_precondition_met "$step_name" "$precondition" || return 0
  echo "== $step_name =="
  LAST_STEP_LOG="$WORK_DIRECTORY/step-${#STEP_NAMES[@]}.log"
  if "$@" >"$LAST_STEP_LOG" 2>&1; then
    record_step "$step_name" "PASS" ""
  else
    tail -n "$FAILURE_LOG_TAIL_LINES" "$LAST_STEP_LOG"
    record_step "$step_name" "FAIL" "command failed"
  fi
}

# Sum of libtest's "running N tests" lines in one test binary's output.
reported_test_count() {
  local log="$1" total=0 count
  while read -r count; do
    total=$((total + count))
  done < <(grep -oP '^running \K[0-9]+' "$log")
  printf '%s' "$total"
}

# run_cargo_test_step <name> <precondition> <package> <features> <test targets>
#                     [extra cargo arguments...]
# The targets are one space-separated string, and each runs in its own cargo
# invocation: a feature-gated test file compiles to an empty binary when its
# feature is missing and passes vacuously, so a target that reported zero tests
# is a FAIL. Running the targets together would hide one empty binary inside the
# others' counts.
run_cargo_test_step() {
  local step_name="$1" precondition="$2" package="$3" features="$4" targets="$5"
  shift 5
  step_precondition_met "$step_name" "$precondition" || return 0
  echo "== $step_name =="
  local step_log="$WORK_DIRECTORY/step-${#STEP_NAMES[@]}.log"
  local target_log="$WORK_DIRECTORY/target.log"
  LAST_STEP_LOG="$step_log"
  : >"$step_log"

  # Every target runs even after one fails: a nightly that stops at the first
  # failure hides the rest until the next night.
  local target target_count=0 total_test_count=0 target_test_count
  local failed_target_count=0 first_failed_target=""
  for target in $targets; do
    target_count=$((target_count + 1))
    if cargo test -p "$package" --features "$features" --test "$target" "$@" \
      >"$target_log" 2>&1; then
      target_test_count="$(reported_test_count "$target_log")"
    else
      target_test_count=-1
    fi
    cat "$target_log" >>"$step_log"
    if [ "$target_test_count" -le 0 ]; then
      if [ "$target_test_count" -eq 0 ]; then
        echo "$target ran 0 tests" | tee -a "$step_log"
      else
        tail -n "$FAILURE_LOG_TAIL_LINES" "$target_log"
      fi
      failed_target_count=$((failed_target_count + 1))
      [ -n "$first_failed_target" ] || first_failed_target="$target"
      continue
    fi
    total_test_count=$((total_test_count + target_test_count))
  done

  if [ "$failed_target_count" -ne 0 ]; then
    record_step "$step_name" "FAIL" \
      "$failed_target_count of $target_count targets failed, first $first_failed_target"
    return
  fi
  record_step "$step_name" "PASS" "$total_test_count tests over $target_count targets"
}

# ---------------------------------------------------------------- preconditions

have_vulkan_device() {
  # Coarse on purpose: it only avoids a long build on a host with no GPU at all.
  # Each vulkan test still skips itself when the adapter lacks video decode.
  if ! compgen -G "/usr/share/vulkan/icd.d/*.json" >/dev/null; then
    SKIP_REASON="no Vulkan ICD"
    return 1
  fi
  if ! compgen -G "/dev/dri/renderD*" >/dev/null; then
    SKIP_REASON="no DRM render node"
    return 1
  fi
}

have_cuda_device() {
  # The driver's control node exists exactly when the NVIDIA driver is loaded.
  if [ ! -c /dev/nvidiactl ]; then
    SKIP_REASON="no NVIDIA driver"
    return 1
  fi
}

have_cuda_device_and_h264_fixture() {
  have_cuda_device || return 1
  if [ -s "$H264_FIXTURE" ]; then
    export G2G_H264_FIXTURE="$H264_FIXTURE"
    return 0
  fi
  if ! command -v ffmpeg >/dev/null 2>&1; then
    SKIP_REASON="no ffmpeg to generate the H.264 fixture"
    return 1
  fi
  mkdir -p "$FIXTURE_DIRECTORY"
  if ! ffmpeg -y -loglevel error \
    -f lavfi -i "testsrc=size=${FIXTURE_WIDTH}x${FIXTURE_HEIGHT}:rate=${FIXTURE_FRAMES_PER_SECOND}:duration=${FIXTURE_DURATION_SECONDS}" \
    -c:v libx264 -pix_fmt yuv420p -g "$FIXTURE_KEYFRAME_INTERVAL" \
    -bsf:v h264_mp4toannexb -f h264 "$H264_FIXTURE"; then
    SKIP_REASON="H.264 fixture generation failed"
    return 1
  fi
  export G2G_H264_FIXTURE="$H264_FIXTURE"
}

have_soak_preconditions() {
  if [ -z "${WAYLAND_DISPLAY:-}" ]; then
    SKIP_REASON="no Wayland session"
    return 1
  fi
  if ! curl --silent --show-error --fail --max-time "$HLS_PROBE_TIMEOUT_SECONDS" \
    --output /dev/null "$HLS_TEST_URL" 2>/dev/null; then
    SKIP_REASON="no HLS feed at $HLS_TEST_URL"
    return 1
  fi
}

have_gstreamer_launch() {
  if ! command -v "${GSTREAMER_LAUNCH%% *}" >/dev/null 2>&1; then
    SKIP_REASON="no $GSTREAMER_LAUNCH"
    return 1
  fi
}

# ---------------------------------------------------------------- suites

vulkan_video_test_targets() {
  local path
  for path in "$REPOSITORY_ROOT"/g2g-plugins/tests/*vulkan*.rs; do
    [ -e "$path" ] || return 1
    basename "$path" .rs
  done
}

run_desktop_gpu_suite() {
  local vulkan_targets
  vulkan_targets="$(vulkan_video_test_targets | tr '\n' ' ')"
  if [ -z "${vulkan_targets// /}" ]; then
    record_step "vulkan video decode" "FAIL" "no vulkan test files found"
  else
    run_cargo_test_step "vulkan video decode" have_vulkan_device \
      g2g-plugins "$VULKAN_VIDEO_FEATURES" "$vulkan_targets"
  fi

  run_cargo_test_step "cuda decode + wgpu bridge" have_cuda_device \
    g2g-plugins "$CUDA_FEATURES" "${CUDA_TEST_TARGETS[*]}"

  run_cargo_test_step "cuda to wgpu end to end" have_cuda_device_and_h264_fixture \
    g2g-ml "$CUDA_WGPU_END_TO_END_FEATURES" cuda_wgpu_e2e

  run_cargo_test_step "A/V lip-sync soak" have_soak_preconditions \
    g2g-plugins "$SOAK_FEATURES" av_lipsync_soak \
    --release -- --ignored --nocapture
}

frames_per_second_for_side() {
  grep -oP "^\s*$2\s.*?fps=\s*\K[0-9.]+" "$1" | head -n 1
}

run_bench_suite() {
  run_step "throughput bench" have_gstreamer_launch \
    bash "$REPOSITORY_ROOT/tools/throughput-bench.sh"
  if [ "$LAST_STEP_RESULT" != "PASS" ]; then
    return
  fi
  local g2g_frames_per_second gstreamer_frames_per_second
  g2g_frames_per_second="$(frames_per_second_for_side "$LAST_STEP_LOG" g2g)"
  gstreamer_frames_per_second="$(frames_per_second_for_side "$LAST_STEP_LOG" gst)"
  set_last_step_detail "g2g fps=${g2g_frames_per_second:-unknown} gst fps=${gstreamer_frames_per_second:-unknown}"
}

# ---------------------------------------------------------------- status

sanitize_description() {
  # Excluding " and \ keeps the payload JSON-safe without an escaper.
  printf '%s' "$1" | tr -cd '[:alnum:] .,:;/=+()%_-' | cut -c "1-$STATUS_DESCRIPTION_MAX_CHARACTERS"
}

build_status_payload() {
  printf '{"state":"%s","context":"%s","description":"%s"}' \
    "$1" "$STATUS_CONTEXT" "$(sanitize_description "$2")"
}

post_commit_status() {
  local state="$1" description="$2"
  local url="$GITHUB_API_BASE/repos/$GITHUB_REPOSITORY/statuses/$TESTED_COMMIT"
  local payload_file="$WORK_DIRECTORY/status-payload.json"
  build_status_payload "$state" "$description" >"$payload_file"

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "dry run, would POST $url"
    cat "$payload_file"
    echo
    return 0
  fi

  local header_file="$WORK_DIRECTORY/authorization-header"
  local body_file="$WORK_DIRECTORY/status-response.json"
  # A builtin printf keeps the token out of any process argument list, and
  # curl reads the header from the file rather than from its command line.
  (umask 077; printf 'Authorization: Bearer %s\n' "$GITHUB_TOKEN" >"$header_file")

  local http_status
  http_status="$(curl --silent --show-error \
    --request POST \
    --header @"$header_file" \
    --header 'Accept: application/vnd.github+json' \
    --header "X-GitHub-Api-Version: $GITHUB_API_VERSION" \
    --data @"$payload_file" \
    --output "$body_file" \
    --write-out '%{http_code}' \
    --max-time "$STATUS_POST_TIMEOUT_SECONDS" \
    "$url")"
  rm -f "$header_file"

  if [ "$http_status" = "$HTTP_CREATED" ]; then
    echo "posted $state to $STATUS_CONTEXT on $TESTED_COMMIT"
    return 0
  fi

  local response_body
  response_body="$(cat "$body_file" 2>/dev/null)"
  if [[ "$response_body" == *"$GITHUB_TOKEN"* ]]; then
    response_body="<withheld: the response echoed the token>"
  fi
  echo "status post failed: HTTP ${http_status:-none}" >&2
  echo "$response_body" >&2
  return 1
}

# ---------------------------------------------------------------- run

if [ "$DRY_RUN" -eq 0 ] && ! read_github_token; then
  exit 1
fi

if ! GITHUB_REPOSITORY="$(github_repository_from_remote)"; then
  exit 1
fi

if ! update_checkout_to_validated_branch; then
  echo "checkout not updated, no status posted" >&2
  exit 1
fi
TESTED_COMMIT="$(git -C "$REPOSITORY_ROOT" rev-parse HEAD)"

echo "== $GITHUB_REPOSITORY $VALIDATED_BRANCH at $TESTED_COMMIT, suite $SUITE_NAME =="

case "$SUITE_NAME" in
  desktop-gpu) run_desktop_gpu_suite ;;
  bench) run_bench_suite ;;
esac

echo
echo "================ host validation report ================"
printf "%-30s %-8s %s\n" "STEP" "RESULT" "DETAIL"
printf "%-30s %-8s %s\n" "----" "------" "------"
for index in "${!STEP_NAMES[@]}"; do
  printf "%-30s %-8s %s\n" "${STEP_NAMES[$index]}" "${STEP_RESULTS[$index]}" "${STEP_DETAILS[$index]}"
done
echo "========================================================"

# A suite that recorded nothing means the runner itself broke, and reporting
# that as success would be the one failure this whole scheme exists to prevent.
if [ "${#STEP_NAMES[@]}" -eq 0 ]; then
  echo "HOST VALIDATION: ERROR (the $SUITE_NAME suite recorded no steps)" >&2
  post_commit_status error "the $SUITE_NAME suite recorded no steps"
  exit 1
fi

STATUS_DESCRIPTION="$PASSED_COUNT passed, $SKIPPED_COUNT skipped, $FAILED_COUNT failed"
for index in "${!STEP_DETAILS[@]}"; do
  case "${STEP_DETAILS[$index]}" in
    *fps=*) STATUS_DESCRIPTION="$STATUS_DESCRIPTION, ${STEP_DETAILS[$index]}" ;;
  esac
done

if [ "$FAILED_COUNT" -ne 0 ]; then
  echo "HOST VALIDATION: FAIL ($STATUS_DESCRIPTION)"
  post_commit_status failure "$STATUS_DESCRIPTION"
  exit 1
fi
echo "HOST VALIDATION: PASS ($STATUS_DESCRIPTION)"
post_commit_status success "$STATUS_DESCRIPTION"
