A barebone implementation of **Vision Transformers** 
([An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)), 
suitable for research. The training loop is based on "[Scaling vision transformers](https://openaccess.thecvf.com/content/CVPR2022/html/Zhai_Scaling_Vision_Transformers_CVPR_2022_paper.html)".
You can load up any of the architectures listed in *Table 2*.

# Setup

Install the dependencies listed in `requirements.txt`:

```shell
pip install -r requirements.txt
```

# Training

Run `train.py` with the following arguments:

```shell
python train.py 
```

# Evaluation

Run `train.py` with the following arguments:

```shell
python eval.py 
```
