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
nvidia-docker build -t occlusion_net .
```

### Setting up data
Follow the steps to download the carfusion dataset and setup at the 
```
git clone https://github.com/dineshreddy91/carfusion_to_coco
cd carfusion_to_coco
virtualenv carfusion2coco -p python3.6
source carfusion2coco/bin/activate
pip install cython numpy
pip install -r requirements.txt
sh carfusion_coco_setup.sh
deactivate
```


### Running with docker


```
sh train.sh occlusion_net
``` 


### Testing on a sample image
Download a pretrained model from  [[Google Drive](https://drive.google.com/open?id=1EUmhzeuMUnv5whv0ZmmOHTbtUiWdeDly)]

Results on a sample demo image

```
sh test.sh occlusion_net demo/demo.jpg
``` 





