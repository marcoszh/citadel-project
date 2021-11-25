PROJECT_DIR=$(pwd)/../
AZURE_STORAGE_CONNECTION_STRING=''
PORT=$1

echo "port: ${PORT}"

docker run -it \
-p ${PORT}:${PORT} \
-e AZURE_STORAGE_CONNECTION_STRING=${AZURE_STORAGE_CONNECTION_STRING} \
-v ${PROJECT_DIR}:/SecureML marcoszh/private_repo:tensorflow \
/bin/bash