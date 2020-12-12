# Detectron2 implementation of DA-RetinaNet [An Unsupervised Domain Adaptation Scheme for Single-Stage Artwork Recognition in Cultural Sites](https://arxiv.org/abs/2008.01882v2) (Image and Vision Computing)
<img src='./Images/DA-RetinaNet.png' width=90%/>

## Introduction
Follow the official guide to install [Detectron2 0.1.1](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) on your pc.

### Data Preparation
If you want to use this code with your dataset arange the dataset in the format of COCO. Inside the script uda_train.py register your dataset using <br> ```register_coco_instances("dataset_name_soruce_training",{},"path_annotations","path_images")```<br>
```register_coco_instances("dataset_name_target_training",{},"path_annotations","path_images")```<br>
```register_coco_instances("dataset_name_target_test",{},"path_annotations","path_images")```<br>

### Training
Replace at the following path ```detectron2/modeling/meta_arch/``` the retinanet.py script with our retinanet.py. Do the same for the fpn.py file at the path ```detectron2/modeling/backbone/```<br>
Run the script train.py

### Testing
If you want to test the model, set to 0 the number of iteration and run the uda_train.py

### Results
<p>
<img src='./Images/results.png' width=42%;/>
</p>

## Citation
Please cite the following [paper](https://arxiv.org/abs/2008.01882) if you use this repository for your project.
```
@misc{pasqualino2020unsupervised,
      title={An Unsupervised Domain Adaptation Scheme for Single-Stage Artwork Recognition in Cultural Sites}, 
      author={Giovanni Pasqualino and Antonino Furnari and Giovanni Signorello and Giovanni Maria Farinella},
      year={2020},
      eprint={2008.01882},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
