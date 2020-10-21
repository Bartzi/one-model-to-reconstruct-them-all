#!/usr/bin/env bash

function usage_and_exit() {
    echo "usage: clean_runs.sh [-y] experiment_folder [experiment_folder...]"
    echo ""
    echo "Options:"
    echo "    -y : confirm yes to deleting old models (you still need to confirm deletion of empty runs)"
    exit 1
}

function check_yes() {
    read -p "$1 [y/N]? " REPLY
    echo ""
    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        return 1
    fi
    return 0
}

function clean_checkpoints()
{
  if ! [ -d "code/wandb" ]; then
    echo "Removing code/wandb..."
    rm -rf "code/wandb"
  fi

  if ! [ -d "checkpoints" ] || [ "$(ls "checkpoints" | wc -l)" = "0" ]; then
    echo "The run does not seem to have any checkpoints. Scanning its size..."
    echo "The folder contains: $(du -sh .)"
    if check_yes "> Delete it COMPLETELY?"; then
      local current_folder="$(basename "$(pwd)")"
      cd ..
      rm -rf "${current_folder}"
    fi
    return
  fi

  # categorize all models
  local all_models=$( cd "checkpoints" && ls )
  local last_snapshot=$( tail -n -1 <<< "${all_models}" | tr '\n' ' ' )

  if [ "${last_snapshot}" = "100000.pt" ] || [ "$((10#${last_snapshot/.pt/}+0))" -le 100000 ]; then
    local old_models=$( head -n -1 <<< "${all_models}" | tr '\n' ' ' )
    local keep_models="${last_snapshot}"
  else
    local keep_models="100000.pt ${last_snapshot}"
    local old_models=$( grep -P -v 100000.pt <<< "${all_models}" | grep -P -v ${last_snapshot} | tr '\n' ' ' )
  fi

  old_models=$( echo "${old_models}" | xargs )
  if [ ! -z "${old_models}" ]; then
    echo "> Deleting the following obsolete model files:"
    echo "    ${old_models}"
    echo "> The following model files are kept:"
    echo "    ${keep_models}"
    if [ "${confirm}" = false ] || check_yes "> Delete the files listed under deletion?"; then
      ( cd "checkpoints" && rm -f -- ${old_models} )
    fi
  fi
}

confirm=true

for arg in "$@"; do
    if [ "${arg}" = "-y" ]; then
        confirm=false
        continue
    fi
done

for argument in "$@"; do
  if [ "${argument}" = "-y" ]; then
    continue
  fi
  (
    echo "Processing ${argument}..."
    cd "${argument}"
    clean_checkpoints
  )
done
