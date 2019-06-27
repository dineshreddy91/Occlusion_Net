BASE=$2
docker run -v $PWD:/code \
        -v $BASE/annotations:/code/datasets/carfusion/annotations \
        -v $BASE/train:/code/datasets/carfusion/train \
        -v $BASE/test:/code/datasets/carfusion/test \
        --shm-size=32GB --runtime=nvidia $1 python train_net.py
        #--shm-size=32GB --runtime=nvidia -it occlusion_net 

