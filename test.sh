docker run -v $PWD:/code --shm-size=32GB --runtime=nvidia $1 python infer.py -ul demo/* -v True -cfg data/occlusion_net_test.yaml

