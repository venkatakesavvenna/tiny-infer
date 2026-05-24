IMAGE_NAME="docgrounding-latest-t"
CONTAINER_NAME="docgrounding-latest-t-container"
CODE_MOUNT="/fsxvision_new/venkat.kesav/backup/tiny-infer"
ENVIRONMENT_MOUNT="/fsxvision_new/venkat.kesav/backup/Environments"

if docker image inspect $IMAGE_NAME >/dev/null 2>&1; then
  echo "Image $IMAGE_NAME already exists."
else
  echo "Building image $IMAGE_NAME..."
  docker build -t $IMAGE_NAME .
fi

if docker ps -aq --filter "name=$CONTAINER_NAME" | grep -q .; then
    echo "Container $CONTAINER_NAME already exists."
    docker exec -d $CONTAINER_NAME bash -c "echo 'Things finally work!'"
else
    docker run --shm-size=512g -dit --gpus all \
        -v $CODE_MOUNT:/code \
        -v $ENVIRONMENT_MOUNT:/environments \
        --name $CONTAINER_NAME \
        -w /code \
        -it \
        $IMAGE_NAME \
        bash -c "/bin/bash"
fi

# --- SSH Setup Section ---
echo "Copying SSH keys into container..."
docker cp ~/.ssh $CONTAINER_NAME:/root/
docker exec $CONTAINER_NAME bash -c "chmod 700 /root/.ssh && chmod 600 /root/.ssh/* && chown -R root:root /root/.ssh"