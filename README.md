# RCN: Residual Compensation Networks for Heterogeneous Face Recognition

Code and testing model from [Zhongying Deng, Xiaojiang Peng, Yu Qiao. “Residual Compensation Networks for Heterogeneous Face Recognition”. AAAI-2019
](https://aaai.org/ojs/index.php/AAAI/article/view/4835)

The experiments is conducted under Caffe and its python interface.

## Training

1. prepare training and testing sets of CASIA NIR-VIS 2.0, then modify the file lists in train_rcn10_NIR_VIS.prototxt
2. add/implement new layers (Normalization layer and CosineLoss layer) to your Caffe
3. train (fine-tune) with instruction like this

```
cd /path/to/your_caffe/
./build/tools/caffe train -solver /path/to/train_rcn10_NIR_VIS.prototxt -gpu 0 -weights /path/to/your_pretrained_model
```

Note:

* The convolutional parameters of pretrained model are frozen, only fully-connected layers learnable
* The learning rates of fc5 layer and fc_adap1/fc6 are different

## Test

1. read the instructions in eval_nirvis2.py and do necessary modifications (e.g. caffe path)
2. run eval_nirvis2.py like this

```
python ./eval_nirvis2.py ./rcn10_NIR_VIS.caffemodel ./deploy.prototxt /path/to/dataset tmp/nirvis2_casianet.pkl
```

Note:

* The alignment of face images should be the same as RCN. Otherwise one could fine-tune rcn10_NIR_VIS.caffemodel on the CASIA NIR-VIS 2.0 with your own aligment.

## Citation

If you use this code, please cite

```
@article{article,
author = {Deng, Zhongying and Peng, Xiaojiang and Qiao, Yu},
year = {2019},
month = {07},
pages = {8239-8246},
title = {Residual Compensation Networks for Heterogeneous Face Recognition},
volume = {33},
journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
doi = {10.1609/aaai.v33i01.33018239}
}
```
