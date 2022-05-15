#/bin/bash
PROJECT_DIRECTORY_PATH="${1}"
BACKUP_TIMESTAMP="${2}"
BACKUP_DIR_PATH="${PROJECT_DIRECTORY_PATH}/backup_${BACKUP_TIMESTAMP}"

rsync -arziv "${BACKUP_DIR_PATH}/" "${PROJECT_DIRECTORY_PATH}"
