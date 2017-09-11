# ResNet with Selu in Tensorflow

ResNets implemented with [Selu activation](https://arxiv.org/abs/1706.02515). The code has been updated to Tensorflow 1.3 and Python 3.5.

I thought that maybe this would provide some performance increase. I was wrong. This network doesn't like to train, but when it does, it only is roughly 50% accurate on CIFAR-10. Once I saw that result, I decided it wasn't worth testing with CIFAR-100 or Imagenet. This was a fun experiment and synthesis of two papers.

You'll notice I left out normalization of all kinds because that is what selu activation is supposed to do if you initialize the weight and bias variables correctly.

If you think I did something wrong and got Selu activation to work well as described in their paper with ResNets, please let me know! I'd be excited to hear about it.

The readme from the original repository is included below. Information there may be inaccurate now, but is kept for the paper link and information about the original repo this code was adapted from.

## ResNet in TensorFlow

Implemenation of [Deep Residual Learning for Image
Recognition](http://arxiv.org/abs/1512.03385).  Includes a tool to use He et
al's published trained Caffe weights in TensorFlow.

MIT license. Contributions welcome.

### Goals

* Be able to use the pre-trained model's that [Kaiming He has provided for
  Caffe](https://github.com/KaimingHe/deep-residual-networks). The `convert.py`
  will convert the weights for use with TensorFlow.

* Implemented in the style of
  [Inception](https://github.com/tensorflow/models/tree/master/inception/inception)
  not using any classes and making heavy use of variable scope. It should be
  easily usable in other models.

* Foundation to experiment with changes to ResNet like [stochastic
  depth](https://arxiv.org/abs/1603.09382), [shared weights at each
  scale](https://arxiv.org/abs/1604.03640), and 1D convolutions for audio. (Not yet implemented.)

* ResNet is fully convolutional and the implementation should allow inputs to be any size.

* Be able to train out of the box on CIFAR-10, 100, and ImageNet. (Implementation incomplete)


### Pretrained Model

To convert the published Caffe pretrained model, run `convert.py`. However
Caffe is annoying to install so I'm providing a download of the output of
convert.py: 

[tensorflow-resnet-pretrained-20160509.tar.gz.torrent](https://raw.githubusercontent.com/ry/tensorflow-resnet/master/data/tensorflow-resnet-pretrained-20160509.tar.gz.torrent)  464M


### Notes

* This code depends on [TensorFlow git commit
  cf7ce8](https://github.com/tensorflow/tensorflow/commit/cf7ce8a7879b6a7ba90441724ea3f8353917a515)
  or later because ResNet needs 1x1 convolutions with stride 2. TF 0.8 is not new
  enough.

* The `convert.py` script checks that activations are similiar to the caffe version
  but it's not exactly the same. This is probably due to differences between how
  TF and Caffe handle padding. Also preprocessing is done with color-channel means 
  instead of pixel-wise means.


