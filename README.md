Occlusion-Net: 2D/3D Occluded Keypoint Localization Using Graph Networks 
======================

[N Dinesh Reddy](http://cs.cmu.edu/~dnarapur), [Minh Vo](http://cs.cmu.edu/~mvo), [Srinivasa G. Narasimhan](http://www.cs.cmu.edu/~srinivas/)

IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 

[[Project](http://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion/)] [[Paper](http://www.cs.cmu.edu/~ILIM/publications/PDFs/RVN-CVPR19.pdf)] [[Supp](http://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion/pdf/occlusion_net_supp.pdf)] [[Bibtex](http://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion/occlusion_net.bib) ]

## Installation

### Setting up with docker

All the stable releases of docker-ce installed from https://docs.docker.com/install/

Install the nvidia-docker from https://github.com/NVIDIA/nvidia-docker

Setting up the docker

```bash
conda install pytorch-nightly cudatoolkit=9.0 -c pytorch
nvidia-docker build -t maskrcnn-benchmark docker/ 
```

### Running with docker
```
docker run -v $PWD:/code -v /media/Car:/media/Car --shm-size=32GB --runtime=nvidia -it maskrcnn-benchmark:latest
``` 



