#!/usr/bin/env bash
# Load HF_TOKEN (and other vars) from .env when not already set in the environment.

export PYTHONPATH="/app/src${PYTHONPATH:+:${PYTHONPATH}}"

load_env_file() {
  local env_file="$1"
  if [[ ! -f "${env_file}" ]]; then
    return 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
  return 0
}

load_env_if_needed() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    return 0
  fi

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  for candidate in \
    "${script_dir}/../.env" \
    "/app/.env" \
    "${PWD}/.env"
  do
    if load_env_file "${candidate}"; then
      echo "Loaded env from ${candidate}"
      return 0
    fi
  done

  return 1
}

require_hf_token() {
  load_env_if_needed || true

  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is required." >&2
    echo "Set it in the environment, in deep-ignorance-extension/.env, or pass --env-file .env to docker run." >&2
    exit 1
  fi
}
