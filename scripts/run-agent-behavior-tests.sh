#!/bin/sh
set -u

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
invocation_dir=$(pwd)
suite_path="$repo_dir/agent-behavior/agent-behavior.json"
models_file="$repo_dir/agent-behavior/models.toml"
models_list="$repo_dir/agent-behavior/models.list"
output_root="$repo_dir/.vyrn/behavior-runs/$(date -u +%Y%m%dT%H%M%SZ)"
selected_models=""
selected_cases=""
keep_workdirs=0

usage() {
  printf '%s\n' \
    "usage: scripts/run-agent-behavior-tests.sh [--case ID] [--model PROFILE] [--output DIR] [--keep-workdirs]" \
    "" \
    "Repeat --case or --model to run a subset. With no filters, every case runs for every profile in agent-behavior/models.list."
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --case)
      [ "$#" -ge 2 ] || { usage >&2; exit 2; }
      selected_cases="${selected_cases}${selected_cases:+
}$2"
      shift 2
      ;;
    --model)
      [ "$#" -ge 2 ] || { usage >&2; exit 2; }
      selected_models="${selected_models}${selected_models:+
}$2"
      shift 2
      ;;
    --output)
      [ "$#" -ge 2 ] || { usage >&2; exit 2; }
      output_root=$2
      shift 2
      ;;
    --keep-workdirs)
      keep_workdirs=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$output_root" in
  /*) ;;
  *) output_root="$invocation_dir/$output_root" ;;
esac

if [ -z "$selected_models" ]; then
  selected_models=$(sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$models_list")
fi

if [ -z "$selected_models" ]; then
  printf 'no behavioral model profiles selected\n' >&2
  exit 2
fi

mkdir -p "$output_root"
temp_root=$(mktemp -d "${TMPDIR:-/tmp}/vyrn-agent-behavior.XXXXXX") || exit 1
cleanup() {
  if [ "$keep_workdirs" -eq 0 ]; then
    rm -rf -- "$temp_root"
  else
    printf 'kept isolated workdirs: %s\n' "$temp_root"
  fi
}
trap cleanup EXIT HUP INT TERM

printf 'building vyrn behavioral-test binary...\n'
if ! cargo build --quiet --manifest-path "$repo_dir/Cargo.toml"; then
  exit 1
fi
vyrn_bin="$repo_dir/target/debug/vyrn"
failed=0

for model_name in $selected_models; do
  [ -n "$model_name" ] || continue
  case "$model_name" in
    *[!A-Za-z0-9._-]*)
      printf 'invalid model profile name for isolated path: %s\n' "$model_name" >&2
      exit 2
      ;;
  esac
  workspace="$temp_root/$model_name"
  mkdir -p "$workspace/.vyrn" "$workspace/home"
  cp "$models_file" "$workspace/.vyrn/models.toml"
  cp -R "$repo_dir/agent-behavior/workspace/." "$workspace/"

  if [ -z "$selected_cases" ]; then
    printf 'running agent behavior suite: model=%s workspace=%s\n' "$model_name" "$workspace"
    if ! (cd "$workspace" && HOME="$workspace/home" "$vyrn_bin" eval "$suite_path" --model "$model_name" --output "$output_root/$model_name"); then
      failed=1
    fi
  else
    for case_id in $selected_cases; do
      case "$case_id" in
        *[!A-Za-z0-9._-]*)
          printf 'invalid case id for isolated path: %s\n' "$case_id" >&2
          exit 2
          ;;
      esac
      printf 'running agent behavior case: model=%s case=%s workspace=%s\n' "$model_name" "$case_id" "$workspace"
      if ! (cd "$workspace" && HOME="$workspace/home" "$vyrn_bin" eval "$suite_path" --model "$model_name" --case "$case_id" --output "$output_root/$model_name/$case_id"); then
        failed=1
      fi
    done
  fi
done

printf 'behavioral traces: %s\n' "$output_root"
exit "$failed"
