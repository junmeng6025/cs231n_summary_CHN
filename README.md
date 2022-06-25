# \[CS231n\] Deep Learning for Computer Vision 学习笔记
参考自https://github.com/mbadry1/CS231n-2017-Summary  
  
  
## 目录
  
  
## 关于课程  
- 网址链接：http://cs231n.stanford.edu/  

- 课程链接：https://www.youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk  

- 教纲链接：http://cs231n.stanford.edu/syllabus.html  

- 作业代码：https://github.com/Burton2000/CS231n-2017  

- 课程章节数： **16**  

- 课程描述：  

  - > Computer Vision has become ubiquitous in our society, with applications in search, image understanding, apps, mapping, medicine, drones, and self-driving cars. Core to many of these applications are visual recognition tasks such as image classification, localization and detection. Recent developments in neural network (aka “deep learning”) approaches have greatly advanced the performance of these state-of-the-art visual recognition systems. This course is a deep dive into details of the deep learning architectures with a focus on learning end-to-end models for these tasks, particularly image classification. During the 10-week course, students will learn to implement, train and debug their own neural networks and gain a detailed understanding of cutting-edge research in computer vision. The final assignment will involve training a multi-million parameter convolutional neural network and applying it on the largest image classification dataset (ImageNet). We will focus on teaching how to set up the problem of image recognition, the learning algorithms (e.g. backpropagation), practical engineering tricks for training and fine-tuning the networks and guide the students through hands-on assignments and a final course project. Much of the background and materials of this course will be drawn from the [ImageNet Challenge](http://image-net.org/challenges/LSVRC/2014/index).  


## 01. Introduction to CNN for visual recognition 卷积神经网络在视觉识别的应用简介  
- 介绍了二十世纪60年代末以来计算机视觉的发展简史；  
- 计算机视觉的主流研究领域：图像分类、物体定位、目标检测、场景理解 等等；  
- [ImageNet](http://www.image-net.org/) 是当今最大的图像分类数据集之一；  
- 自2012年以来，在基于 ImageNet 的图像识别挑战当中，CNN（卷积神经网络）一直展现出强悍的性能；  
- CNN 是法国计算机科学家 [杨乐昆](https://en.wikipedia.org/wiki/Yann_LeCun?wprov=sfla1) 于1997年[首次提出](http://ieeexplore.ieee.org/document/726791/)  
  
## 02. Image Classification 图像分类  
- 图像分类领域有许多富有挑战性的问题，例如：光照 (illumination)、视角变化 (view point)、遮挡 (occlusion) 等等；  
- 图像分类可以采取 kNN 算法 (k nearest neighborhood)，但处理效果较差。  
- kNN 算法的超参数：  
  - k 值：在对象点附近用于对比的样本数；  
  - distance measurement 距离度量：  
    - L2 距离（欧几里得距离，即直线段距离）：适用于无坐标属性的点；  
    - L1 距离（曼哈顿距离，即棋盘格距离）：适用于有坐标属性的点；  
- 可以使用**交叉验证 (Cross Validation)** 的方法来优化超参数（此处我们尝试用 tp 预测 K）  
  1) 把数据集分成 `f` 等份子数据集  
  2) 给定一组超参数，用 `f-1` 份做训练，剩下的 `1` 份做测试。  
  3) 所有子数据集训练效果的均值作为该组超参数的训练效果。
  4) 更换一组超参数，用与之前不同的 `f-1` 份做训练，剩下的 `1` 份做测试。  
  5) 重复 `f` 次，使每一份都被用作过测试集  
  6) 选择训练效果最好的超参数作为最终的模型参数。

