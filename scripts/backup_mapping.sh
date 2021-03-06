#!/bin/bash
PROJECT_DIRECTORY_PATH="${1}"
DATE="$(date +%s)"
NAME="${2-}"
if [[ -n "${NAME}" ]]; then
  NAME="-${NAME}"
fi

BACKUP_DIR_PATH="${PROJECT_DIRECTORY_PATH}/backup-${DATE}${NAME}"
mkdir ${BACKUP_DIR_PATH}

declare -a include_array=(  /
  "*.yaml"                  /
  "expt_record.z"           /
  "correspondences/*.npy"   /
  "clusterings/*.npy"       /
  "embeddings/*.npy"        /
  )
for included_files in "${include_array[@]}"
do
  rsync -marziv --exclude="*backup*" --include="*/" --include="${included_files}" --exclude="*" "${PROJECT_DIRECTORY_PATH}/" "${BACKUP_DIR_PATH}"
done
echo ${DATE}

rsync -marziv "${PROJECT_DIRECTORY_PATH}/results" "${BACKUP_DIR_PATH}"
