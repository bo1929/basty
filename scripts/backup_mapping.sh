#!/bin/bash
PROJECT_DIRECTORY_PATH="${1}"
BACKUP_DIR_PATH="${PROJECT_DIRECTORY_PATH}/backup_$(date +%s)"
mkdir ${BACKUP_DIR_PATH}

declare -a include_array=( \ 
  "*.yaml" "correspondences/*.npy" "clusterings/*.npy" "embeddings/*.npy" "expt_record.z" \ 
)
for included_files in "${include_array[@]}"
do
  rsync -marziv --exclude="*backup*" --include="*/" --include="${included_files}" --exclude="*" "${PROJECT_DIRECTORY_PATH}/" "${BACKUP_DIR_PATH}"
done
