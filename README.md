![ViT](https://user-images.githubusercontent.com/26739999/142579081-b5718032-6581-472b-8037-ea66aaa9e278.png)

A barebone implementation of **Vision Transformers** 
([An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)), 
suitable for research. The training loop as well as the default hyperparameter values are based on 
"[Scaling vision transformers](https://openaccess.thecvf.com/content/CVPR2022/html/Zhai_Scaling_Vision_Transformers_CVPR_2022_paper.html)",
except that this code does not perform *MixUp*. The authors recommend to use MixUp (level 10) with a 20% probability.

# Setup

Install the dependencies listed in `requirements.txt`:

```shell
pip install -r requirements.txt
```

# Architectures

You can load up any of the architectures listed in *Table 2*. Note that these hyperparameters are intended for the 
*ImageNet-21k* dataset, and are to be used with 256x256 images and 1024-sample batches.

![Architectures](https://d3i71xaburhd42.cloudfront.net/2a805d0e1b067444a554c5169d189fa1f649f411/7-Table2-1.png)

# Data

The training/evaluation scripts will first look for your dataset locally, and if they fail to find it, they will look
in the [`torchvision.datasets`](https://pytorch.org/vision/stable/datasets.html#image-classification) module.

Local datasets must be accessible via the path passed to the `--data-dir` argument, which defaults to the `data` 
directory. Within each folder associated with a local dataset, there must be separate folders containing `train`/`val`/`test` 
splits. You can override these names using optional arguments. Furthermore, each split must follow a structure that the 
[`ImageFolder`](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) class comprehends.

Datasets from the [`torchvision.datasets`](https://pytorch.org/vision/stable/datasets.html#image-classification) module
must accept `train` and `transform` as arguments, *which limits your options a little bit!*

# Training

Run `train.py`, passing in the name of your target architecture, the name of the directory containing your dataset,
and a name for the experiment, which will be used for creating the checkpoints and storing the results.

Example:

```shell
python train.py s/16 CIFAR-10 alpha
```

You can pass in the optional `--cont` argument, which will try to load the latest checkpoint and continue the
training process. Note that if no previous checkpoint exists, the code will not raise an exception and will simply
start training from scratch.

There are other optional arguments, as well:

``` 
  --cpu-only            When passed in, the GPU will not be used.
  --no-augment          When passed in, the recommended augmentations will not be applied.
  --no-grad-clip        When passed in, the gradients will not be clipped before the optimization step.
  --max-grad-norm       The `max_norm` argument passed to `clip_grad_norm_`. Defaults to 1.0.
  --learning-rate       Adam's learning rate. Defaults to 1e-3.
  --weight-decay        Adam's weight decay. Defaults to 1e-4.
  --num-epochs          Number of epochs. Defaults to 90.
  --image-size          Target size of the images. Defaults to 256.
  --batch-size          Size of the batches. Defaults to 1024.
  --num-classes         Number of classes. Defaults to 1000.
  --warmup-iters        How many iterations to wait before annealing the learning rate. Defaults to 10000.
  --logging-iters       How often the loss and metrics are written to TensorBoard. Defaults to 50 iterations. 
  --checkpoint-iters    How often the model (and other state dicts) are stored. Defaults to 1000 iterations.
  --checkpoints-keep    How many checkpoints to keep during training. Defaults to 5.
  --checkpoints-dir     Where to store the checkpoint files. Defaults to "checkpoints".
  --experiments-dir     Where to store the experiments' results. Defaults to "experiments".
  --log-dir             Target folder for writing TensorBoard logs. Defaults to "logs".
  --data-dir            Path to the folder containing your locally-stored datasets. Defaults to "data".
  --train-split         Name of the subfolder containing the training split. Used for local datasets. Defaults to "train".
  --val-split           Name of the subfolder containing the training split. Used for local datasets. Defaults to "val".
```

# Evaluation

Run `eval.py`, passing in the name of your target dataset and the name you chose for your experiment during training.

Example:

```shell
python eval.py CIFAR-10 alpha 
```

There are also several optional arguments:

```
  TO DO
```

# Results

*To be completed.*

You'll be able to find the trained models in the **Releases** section.
