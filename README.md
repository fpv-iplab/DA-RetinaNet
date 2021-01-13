# Detectron2 implementation of DA-RetinaNet [An Unsupervised Domain Adaptation Scheme for Single-Stage Artwork Recognition in Cultural Sites](https://arxiv.org/abs/2008.01882v3) (Image and Vision Computing 2021)
<img src='./Images/DA-RetinaNet.png' width=90%/>

## Introduction
Follow the official guide to install [Detectron2 0.2.1](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) on your pc.

### Data Preparation
If you want to use this code with your dataset arrange the dataset in the format of COCO. Inside the script uda_train.py register your dataset using <br> ```register_coco_instances("dataset_name_soruce_training",{},"path_annotations","path_images")```<br>
```register_coco_instances("dataset_name_target_training",{},"path_annotations","path_images")```<br>
```register_coco_instances("dataset_name_target_test",{},"path_annotations","path_images")```<br>

### Training
Replace at the following path ```detectron2/modeling/meta_arch/``` the retinanet.py script with our retinanet.py. Do the same for the fpn.py file at the path ```detectron2/modeling/backbone/```<br>
Run the script uda_train.py <br>
Trained model is available at this links: <br>
[DA-RetinaNet](https://iplab.dmi.unict.it/EGO-CH-OBJ-UDA/DA-RetinaNet.pth) <br>
[DA-RetinaNet-CycleGAN](https://iplab.dmi.unict.it/EGO-CH-OBJ-UDA/DA-RetinaNet-CycleGAN.pth) <br>

### Testing
If you want to test the model load the new weights, set to 0 the number of iterations and run the uda_train.py

### Results
<p>
  Results of DA-Faster RCNN, Strong-Weak and the proposed DA-RetinaNet combined with image-to-image translation approach.
</p>

<table style="width:100%">
  <tr>
    <th></th>
    <th colspan="2">image to image translation (CycleGAN)</th>
  </tr>
  <tr>
    <td>Object Detector</td>
    <td>None</td>
    <td>Synthetic to Real</td>
  </tr>
   <tr>
    <td>DA-Faster RCNN</td>
    <td>12.94%</td>
    <td>33.20%</td>
  </tr>
   <tr>
    <td>StrongWeak</td>
    <td>25.12%</td>
    <td>47.70%</td>
  </tr>
  <tr>
    <td>DA-RetinaNet</td>
    <td>31.04%</td>
    <td>58.01%</td>
  </tr>
</table>


## Citation
Please cite the following [paper](https://arxiv.org/abs/2008.01882) if you use this repository for your project.
```
@article{PASQUALINO2021104098,
  title = "An unsupervised domain adaptation scheme for single-stage artwork recognition in cultural sites",
  journal = "Image and Vision Computing",
  pages = "104098",
  year = "2021",
  issn = "0262-8856",
  doi = "https://doi.org/10.1016/j.imavis.2021.104098",
  url = "http://www.sciencedirect.com/science/article/pii/S0262885621000032",
  author = "Giovanni Pasqualino and Antonino Furnari and Giovanni Signorello and Giovanni Maria Farinella",
  keywords = "Object detection, Cultural sites, First person vision, Unsupervised domain adaptation",
}
```
## Other Works
[MDA-RetinaNet](https://github.com/fpv-iplab/MDA-RetinaNet)
