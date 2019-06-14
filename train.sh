BASE=$PWD/carfusion_to_coco/datasets/carfusion/

docker run -v $PWD:/code \
        -v $BASE/annotations/car_keypoints_train.json:/code/datasets/carfusion/annotations/car_keypoints_train.json \
        -v $BASE/annotations/car_keypoints_test.json:/code/datasets/carfusion/annotations/car_keypoints_test.json \
        -v $BASE/train:/code/datasets/carfusion/train \
        -v $BASE/test:/code/datasets/carfusion/test \
        --shm-size=32GB --runtime=nvidia $1 python train_net.py

