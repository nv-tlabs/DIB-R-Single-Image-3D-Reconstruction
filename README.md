This is the training code for:

#### Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer (NeurIPS 2019)

[Wenzheng Chen](http://www.cs.toronto.edu/~wenzheng/), [Jun Gao\*](http://www.cs.toronto.edu/~jungao/), [Huan Ling\*](http://www.cs.toronto.edu/~linghuan/), [Edward J. Smith\*](), [Jaakko Lehtinen](https://users.aalto.fi/~lehtinj7/), [Alec Jacobson](https://www.cs.toronto.edu/~jacobson/), [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)


**[[Paper](https://arxiv.org/abs/1908.01210)]  [[Project Page](https://nv-tlabs.github.io/DIB-R/)]**

## Usage


# requirements
We recommneded to use conda to install all the dependencies.

Tested under python3.7.10, cuda 11.1, kaolin0.9.0

```
# pytorch
conda create --name dibr python=3.7.10
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
# other libararies
pip install opencv-python tensorboardX  
# import render from kaolin since it is optimized
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
python setup.py develop
cd ..
```

# dataset
To better reproduce the model, we use the public rendered images of shapenet from DVR repo https://github.com/autonomousvision/differentiable_volumetric_rendering.
Note it is pretty big (31.5GB).
Please downlaod the choice 2 to get the renderings for 2D supervised models.

# train
```
git clone repo
cd repo
cd train
python train.py --datafolder YOUR_DATASET_FOLDER  --svfolder YOUR_SAVING_FOLDER
```

# eval
```
python eval.py --datafolder YOUR_DATASET_FOLDER  --svfolder YOUR_SAVING_FOLDER --iterbe YOUR_MODE_ITER --viewnum 1
cd ../chamfer
python check_chamfer.py  --folder YOUR_SAVING_FOLDER --gt_folder YOUR_DATASET_FOLDER 
```

# performance

tested with chckpoint 469999 iterations.

|  Class | 02691156| 02828884|02933112|02958343|03001627|03211117|03636649|03691459|04090263|04256520|04379243|04401088|04530566 | Average |
|  ----  | ----  |     ----  |   ----  |  ----  | ----  |----  |  ----  |   ----  |   ----  |  ----  | ----  |----  |   ----  |  ----  |
| Scores  | 1.9011142659 | 2.4427891617 | 2.9287128952| 2.8782509213| 3.7478455647| 3.2151179278| 4.0958526133| 3.6632570197| 1.8829994567| 3.2710785272| 3.1060515590| 2.1552766258| 2.6679806140| 2.9197174732 |


## Citations

Please consider the following citation if you find our code is useful: 

```
@inproceedings{chen2019_dibr,
 author = {Chen, Wenzheng and Ling, Huan and Gao, Jun and Smith, Edward and Lehtinen, Jaakko and Jacobson, Alec and Fidler, Sanja},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer},
 url = {https://proceedings.neurips.cc/paper/2019/file/f5ac21cd0ef1b88e9848571aeb53551a-Paper.pdf},
 volume = {32},
 year = {2019}
}

```
