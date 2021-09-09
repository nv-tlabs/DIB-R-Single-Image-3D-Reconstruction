
This is the code release for DIB-R training,compared to its official 


# requirements
tested under python3.7.10, cuda 11.1, kaolin
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
We directy use the rendered shapenet from DVR repo.
Note it is pretty big (31.5GB).
```
wget https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip
unzip NMR_Dataset.zip
```

# train
```
# download
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



