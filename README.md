Semi-Supervised Anomaly Detection Based on Deep Generative Models with Transformer
====
source code of Semi-Supervised Anomaly Detection Based on Deep Generative Models with Transformer
----
Requirements
----
```python
pyhton >= 3.7.3
torch >= 1.7.0
torchvision >= 0.8.1
einops >= 0.3.0
```
Training on CIFAR10
-----
To train the model on CIFAR10 dataset for a given anomaly class, run the following:

```python
python train.py
    --dataset cifar10                                                   
    --niter <number-of-epochs>                                          
    --abnormal_class                                                    
        <plane, car, bird, cat, deer, dog, frog, horse, ship, truck>    
    --display                       # optional if you want to visualize        
```

Train on Custom Dataset
----
To train the model on a custom dataset, the dataset should be copied into `./data` directory, and should have the following directory & file structure:

```python
Custom Dataset
├── test
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_n.png
│   ├── 1.abnormal
│   │   └── abnormal_tst_img_0.png
│   │   └── abnormal_tst_img_1.png
│   │   ...
│   │   └── abnormal_tst_img_m.png
├── train
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_t.png

```
