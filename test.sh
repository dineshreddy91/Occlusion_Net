docker run -v $PWD:/code --shm-size=32GB --runtime=nvidia $1 python python infer.py -ul demo/* -v True -cfg occlusion_net.yaml

