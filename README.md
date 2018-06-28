# YOLO from scratch

Learn & build CNN based on YOLO implementation

- [1. overview](./docs/01_overview.md)
- [2. preprocess](./docs/02_preprocess.md)
- [3. model](./docs/03_network.md)
- [4. postprocess](./docs/04_postprocess.md)

## Prepare Dataset

1. Download dataset from [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
2. Download labels from [Link]() or parsed by [voc_label.py](https://pjreddie.com/media/files/voc_label.py)

## CPU verion
```bash
cd docker
docker-compose up -d
```
Navigate to [localhost:8888](localhost:8888)

## GPU version
Install cuda-toolkit in host machine([reference](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation)), and then
```bash
cd docker
docker-compose -f docker-compose-gpu.yml up -d # this command would build a new docker image in the first time
```
Navigate to [localhost:8888](localhost:8888), password is defined as `PASSWORD` environment variable in `docker-compose-gcp.yml`

## Weights

## Todos
- [x] preprocess image: scaling and padding to 448x448
- [x] preprocess labels: convert to cell(grid) coordinate
- [x] prepare the docker image for tensorflow-gpu
- [x] tiny network: reference from [onther implementation](https://github.com/persistforever/yolo-tensorflow)
- [x] implement loss function based on YOLO paper
- [x] postprocess: based on [Coursera CNN](https://www.coursera.org/learn/convolutional-neural-networks/home/welcome)
- [x] document:
- [x] data argumentation: Flip
- [ ] cross validation
- [ ] data argumentation: HSV transform
- [ ] deepest neural network
- [ ] YOLO3 anchor box and loss function

## Reference
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)
- [Darknet](https://pjreddie.com/darknet/yolo/)
- [Andrew Ng's CNN course](https://www.coursera.org/learn/convolutional-neural-networks/home/welcome)
- [persistforever/yolo-tensorflow](https://github.com/persistforever/yolo-tensorflow)
