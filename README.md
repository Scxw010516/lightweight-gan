<img src="./images/sample-512.jpg" width="600px"></img>

*512x512 flowers after 12 hours of training, 1 gpu*

<img src="./images/sample-256.jpg" width="400px"></img>

*256x256 flowers after 12 hours of training, 1 gpu*

<img src="./images/pizza-512.jpg" width="600px"></img>

*Pizza*

## 'Lightweight' GAN

[![PyPI version](https://badge.fury.io/py/lightweight-gan.svg)](https://badge.fury.io/py/lightweight-gan)

2021在ICLR中提出的“轻量级”GAN(Pytorch版)。本文的主要贡献是生成器中的跳跃层激励，以及鉴别器中的自动编码自监督学习。引用一行总结“在1024张分辨率低于100张的图像上，通过几个小时的训练，在单个gpu上收敛”。

## Install

```bash
$ pip install lightweight-gan
```

## Use

一行命令

```bash
$ lightweight_gan --data ./path/to/images --image-size 512
```

每1000次迭代，模型将保存到“.models｛name｝”，模型中的样本保存到“.results｛name}”。默认情况下，“name”将为“default”。

## Training settings训练设置

对于深度学习从业者来说，这是不言自明的

```bash
$ lightweight_gan \
    --data ./path/to/images \
    --name {name of run} \
    --batch-size 16 \
    --gradient-accumulate-every 4 \
    --num-train-steps 200000
```

## Augmentation增强

增强对于轻量级GAN在低数据环境中有效工作至关重要

默认情况下，增强类型设置为平移和剪切，并省略颜色。您也可以将颜色包含在以下内容中。

```bash
$ lightweight_gan --data ./path/to/images --aug-prob 0.25 --aug-types [translation,cutout,color]
```

### Test augmentation测试增强

你可以在图像进入神经网络之前测试并查看图像将如何增强（如果你使用增强）。让我们看看它是如何处理此图像的：

![](./docs/aug_test/lena.jpg)

#### Basic usage基本使用

增强图像、定义`--aug test`并将图像路径放入`--data`的基本代码：

```bash
lightweight_gan \
    --aug-test \
    --data ./path/to/lena.jpg
```

创建后，文件lena_augs.jpg将如下所示：

![](./docs/aug_test/lena_augs_default.jpg)


#### Options选项

您可以使用一些选项来更改结果：
- `--image-size 256` 改变`image-size`. 默认: `256`.
- `--aug-type [color,cutout,translation]` 组合几个增强功能. 默认: `[cutout,translation]`.
- `--batch-size 10` 更改`batch-size`. 默认: `10`.
- `--num-image-tiles 5` 更改`num-image-tiles`. 默认: `5`.

尝试此命令：
```bash
lightweight_gan \
    --aug-test \
    --data ./path/to/lena.jpg \
    --batch-size 16 \
    --num-image-tiles 4 \
    --aug-types [color,translation]
```

结果会是这样的：

![](./docs/aug_test/lena_augs.jpg)

### Types of augmentations扩增类型

这个库包含几种类型的嵌入式扩充.  

其中一些默认情况下有效，其中一些可以通过命令`--aug-types`的选项进行控制:
- 水平翻转（默认情况下工作，不受控制，在AugWrapper类中运行）;
- `color` 随机改变亮度、饱和度和对比度;
- `cutout` 在图像上创建随机的黑框; 
- `offset` 使用重复图像按x轴和y轴随机移动图像;
  - `offset_h` only by an x-axis;
  - `offset_v` only by a y-axis;
- `translation` 在画布上随机移动黑色背景的图像;

增强功能的完整设置 `--aug-types [color,cutout,offset,translation]`.  
一般建议为您的数据使用合适的aug，并尽可能多地使用，然后在训练一段时间后禁用最具破坏性的（图像）aug。

#### Color

![](./docs/aug_types/lena_augs_color.jpg)

#### Cutout

![](./docs/aug_types/lena_augs_cutout.jpg)

#### Offset

![](./docs/aug_types/lena_augs_offset.jpg)

Only x-axis:

![](./docs/aug_types/lena_augs_offset_h.jpg)

Only y-axis:

![](./docs/aug_types/lena_augs_offset_v.jpg)

#### Translation

![](./docs/aug_types/lena_augs_translation.jpg)

## Mixed precision混合精度

你可以用`--amp`打开自动混合精度

你应该期望它能快33%，并节省高达40%的内存

## Multiple GPUs多GPUs

设置项`--multi-gpus`

## Visualizing training insights with Aim使用Aim可视化训练

[Aim](https://github.com/aimhubio/aim) 是一个开源的实验跟踪器，它记录你的训练运行，使用一个漂亮的UI来比较它们，并使用一个API以编程方式查询它们。

```bash
$ pip install aim
```

接下来，您可以使用`--Aim_repo`标志指定Aim日志目录，否则日志将存储在当前目录中

```bash
$ lightweight_gan --data ./path/to/images --image-size 512 --use-aim --aim_repo ./path/to/logs/
```

执行`aim up --repo ./path/to/logs/`在服务器上运行Aim UI。

**在仪表板中查看所有追踪的运行、每个指标在上次追踪的值和追踪的超参数:**

<img width="1431" alt="Screen Shot 2022-04-19 at 00 48 55" src="https://user-images.githubusercontent.com/11066664/163875698-dc497334-1f77-4e18-a37e-ac0f874b9814.png">


**使用Metrics Explorer比较损失曲线-通过任何超参数分组和聚合，轻松比较运行情况:**

<img width="1440" alt="Screen Shot 2022-04-12 at 16 56 35" src="https://user-images.githubusercontent.com/11066664/163875452-1da3bf36-f3bc-449f-906e-cebaf9a4fd6c.png">

**跨训练步骤比较和调试生成的图像，并通过图像资源管理器运行:**

<img width="1439" alt="Screen Shot 2022-04-12 at 16 57 24" src="https://user-images.githubusercontent.com/11066664/163875815-9cd8ce85-2815-4f0a-80dd-0f3258193c19.png">

## Generating生成

完成训练后，可以使用一个命令生成样本。您可以选择要从中加载的检查点编号。如果未指定`--load-from`，则默认为最新版本。

```bash
$ lightweight_gan \
  --name {name of run} \
  --load-from {checkpoint num} \
  --generate \
  --generate-types {types of result, default: [default,ema]} \
  --num-image-tiles {count of image result}
```

运行此命令后，您将获得结果图像文件夹附近的文件夹，该文件夹带有后缀"-generated-{checkpoint num}"。

也可以生成插值

```bash
$ lightweight_gan --name {name of run} --generate-interpolation
```

## Show progress显示进度

在创建了几个模型检查点后，您可以通过命令将进度生成为序列图像：

```bash
$ lightweight_gan \
  --name {name of run} \
  --show-progress \
  --generate-types {types of result, default: [default,ema]} \
  --num-image-tiles {count of image result}
```

运行此命令后，您将在results文件夹中获得一个新文件夹，后缀为“-progress”。您可以使用命令`ffmpeg -framerate 10 -pattern_type glob -i '*-ema.jpg' out.mp4`"`将图像转换为带有ffmpeg的视频。

![Show progress gif demonstration](./docs/show_progress/show-progress.gif)

![Show progress video demonstration](./docs/show_progress/show-progress.mp4)

## Discriminator output size鉴别器输出大小

作者好心地告诉我，鉴别器输出大小（5x5 vs 1x1）在不同的数据集上会导致不同的结果。（举个例子，5x5对艺术比对人脸更有效）。您可以使用单个标志进行切换

```bash
# disc output size is by default 1x1
$ lightweight_gan --data ./path/to/art --image-size 512 --disc-output-size 5
```

## Attention注意

您可以使用以下方法将线性+轴向注意力添加到特定分辨率的图层

```bash
# make sure there are no spaces between the values within the brackets []
$ lightweight_gan --data ./path/to/images --image-size 512 --attn-res-layers [32,64] --aug-prob 0.25
```

## Dual Contrastive Loss双重对比度损失

A recent paper has proposed that a novel contrastive loss between the real and fake logits can improve quality slightly over the default hinge loss.

You can use this with one extra flag as follows

最近的一篇论文提出，真实和虚假logits之间的新的对比损失可以比默认的铰链损失稍微提高质量。

您可以将其与一个额外的标志一起使用，如下所示

```bash
$ lightweight_gan --data ./path/to/images --dual-contrast-loss
```

## Bonus额外

您也可以使用透明图像进行训练

```bash
$ lightweight_gan --data ./path/to/images --transparent
```

或灰度

```bash
$ lightweight_gan --data ./path/to/images --greyscale
```

## Alternatives选择

如果你想要最新的art GAN，你可以在 https://github.com/lucidrains/stylegan2-pytorch

## Citations引用

```bibtex
@inproceedings{
    anonymous2021towards,
    title   = {Towards Faster and Stabilized {\{}GAN{\}} Training for High-fidelity Few-shot Image Synthesis},
    author  = {Anonymous},
    booktitle = {Submitted to International Conference on Learning Representations},
    year    = {2021},
    url     = {https://openreview.net/forum?id=1Fqg133qRaI},
    note    = {under review}
}
```

```bibtex
@misc{cao2020global,
    title   = {Global Context Networks},
    author  = {Yue Cao and Jiarui Xu and Stephen Lin and Fangyun Wei and Han Hu},
    year    = {2020},
    eprint  = {2012.13375},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{qin2020fcanet,
    title   = {FcaNet: Frequency Channel Attention Networks},
    author  = {Zequn Qin and Pengyi Zhang and Fei Wu and Xi Li},
    year    = {2020},
    eprint  = {2012.11879},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{yu2021dual,
    title   = {Dual Contrastive Loss and Attention for GANs}, 
    author  = {Ning Yu and Guilin Liu and Aysegul Dundar and Andrew Tao and Bryan Catanzaro and Larry Davis and Mario Fritz},
    year    = {2021},
    eprint  = {2103.16748},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@article{Sunkara2022NoMS,
    title   = {No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects},
    author  = {Raja Sunkara and Tie Luo},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.03641}
}
```

