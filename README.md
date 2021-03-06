
论文原文:https://arxiv.org/pdf/1409.1556.pdf

主要参考自[csdn](https://blog.csdn.net/DreamBro/article/details/121068023)

注意到vgg论文提到测试时使用1x1卷积核代替全连接

pytorch 1x1卷积核代替全连接参考:https://discuss.pytorch.org/t/vgg-with-1x1-convolution/57980

1x1代替全连接数学推导:https://datascience.stackexchange.com/questions/12830/how-are-1x1-convolutions-the-same-as-a-fully-connected-layer

pytorch.argmax:https://blog.csdn.net/artistkeepmonkey/article/details/115067766

代码中用到的数据集（直接解压到项目文件夹）下载:https://pan.baidu.com/s/1rPZzQTE00r8lnc9Ott9j2Q?pwd=n5im


代码中的网络结构如下
```
class VGG(nn.Module):
    def __init__(self,num_classes=2):                                       #分类数量,原论文为1000,输入为224z224
        super(VGG, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1,padding= 1),           # same
            nn.ReLU(inplace=True),                                          #True代表传入值将被改变（引用赋值）
            nn.Conv2d(64, 64,kernel_size= 3,stride =1,padding= 1),          #same
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),                           #(65-2)/2+1=32 

            # 第二层
            nn.Conv2d(64, 128, kernel_size=3,stride= 1, padding=1),         #same
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128,kernel_size= 3,stride= 1,padding= 1),        #same
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride= 2),                          #(32-2)/2+1=16

            # 第三层
            nn.Conv2d(128, 256, kernel_size=3, stride=1,padding= 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding= 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256,kernel_size= 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),                           #(16-2)/2+1=8

            # 第四层
            nn.Conv2d(256, 512,kernel_size= 3,stride= 1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride= 1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride= 1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                          #(8-2)/2+1=4

            # 第五层
            nn.Conv2d(512, 512,kernel_size= 3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512,kernel_size= 3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride= 1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)                                               #(4-2)/2+1=2
            )
        self.classifier=nn.Sequential(
            nn.Linear(512*2*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
            #nn.Conv2d(in_channels=4096, out_channels=n_classes, kernel_size=1), #VGG论文中提到，用1x1卷积核代替全连接
            )

 
 
    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,start_dim=1)                                      #展开，进入全连接层
        x=self.classifier(x)
 
        return x

```

# VGG神经网络结构



vgg神经网络不同配置的结构如图所示

![constructure](https://user-images.githubusercontent.com/74494790/170701867-7d9c64b5-8fb1-4966-aed3-3110618666b6.png)


以C类为例

1、输入224x224x3的图片，经64个3x3的卷积核作两次卷积+ReLU，卷积后的尺寸变为224x224x64

2、作max pooling（最大化池化），池化单元尺寸为2x2（效果为图像尺寸减半），池化后的尺寸变为112x112x64

3、经128个3x3的卷积核作两次卷积+ReLU，尺寸变为112x112x128

4、作2x2的max pooling池化，尺寸变为56x56x128

5、经256个3x3的卷积核作三次卷积+ReLU，尺寸变为56x56x256

6、作2x2的max pooling池化，尺寸变为28x28x256

7、经512个3x3的卷积核作三次卷积+ReLU，尺寸变为28x28x512

8、作2x2的max pooling池化，尺寸变为14x14x512

9、经512个3x3的卷积核作三次卷积+ReLU，尺寸变为14x14x512

10、作2x2的max pooling池化，尺寸变为7x7x512

11、与两层1x1x4096，一层1x1x1000进行全连接+ReLU（共三层）

12、通过softmax输出1000个预测结果

本代码实现的VGG结构
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 65, 65]           1,792
              ReLU-2           [-1, 64, 65, 65]               0
            Conv2d-3           [-1, 64, 65, 65]          36,928
              ReLU-4           [-1, 64, 65, 65]               0
         MaxPool2d-5           [-1, 64, 32, 32]               0
            Conv2d-6          [-1, 128, 32, 32]          73,856
              ReLU-7          [-1, 128, 32, 32]               0
            Conv2d-8          [-1, 128, 32, 32]         147,584
              ReLU-9          [-1, 128, 32, 32]               0
        MaxPool2d-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 256, 16, 16]         295,168
             ReLU-12          [-1, 256, 16, 16]               0
           Conv2d-13          [-1, 256, 16, 16]         590,080
             ReLU-14          [-1, 256, 16, 16]               0
           Conv2d-15          [-1, 256, 16, 16]         590,080
             ReLU-16          [-1, 256, 16, 16]               0
        MaxPool2d-17            [-1, 256, 8, 8]               0
           Conv2d-18            [-1, 512, 8, 8]       1,180,160
             ReLU-19            [-1, 512, 8, 8]               0
           Conv2d-20            [-1, 512, 8, 8]       2,359,808
             ReLU-21            [-1, 512, 8, 8]               0
           Conv2d-22            [-1, 512, 8, 8]       2,359,808
             ReLU-23            [-1, 512, 8, 8]               0
        MaxPool2d-24            [-1, 512, 4, 4]               0
           Conv2d-25            [-1, 512, 4, 4]       2,359,808
             ReLU-26            [-1, 512, 4, 4]               0
           Conv2d-27            [-1, 512, 4, 4]       2,359,808
             ReLU-28            [-1, 512, 4, 4]               0
           Conv2d-29            [-1, 512, 4, 4]       2,359,808
             ReLU-30            [-1, 512, 4, 4]               0
        MaxPool2d-31            [-1, 512, 2, 2]               0
           Linear-32                 [-1, 4096]       8,392,704
             ReLU-33                 [-1, 4096]               0
          Dropout-34                 [-1, 4096]               0
           Linear-35                 [-1, 4096]      16,781,312
             ReLU-36                 [-1, 4096]               0
          Dropout-37                 [-1, 4096]               0
           Linear-38                    [-1, 2]           8,194
================================================================
Total params: 39,896,898
Trainable params: 39,896,898
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 18.27
Params size (MB): 152.19
Estimated Total Size (MB): 170.51
----------------------------------------------------------------

```

# VGG网络创新点

1. 结构简洁。VGG由5层卷积层、3层全连接层、softmax输出层构成，层与层之间使用max-pooling分开，所有隐层的激活单元都采用ReLU函数。
2. 小卷积核和多卷积子层。VGG使用多个较小卷积核（3x3）的卷积层代替一个卷积核较大的卷积层，一方面可以减少参数，另一方面相当于进行了更多的非线性映射，可以增加网络的拟合/表达能力。VGG通过降低卷积核的大小（3x3），增加卷积子层数来达到同样的性能。
3. 小池化核。相比AlexNet的3x3的池化核，VGG全部采用2x2的池化核。
4. 通道数多。VGG网络第一层的通道数为64，后面每层都进行了翻倍，最多到512个通道，通道数的增加，使得更多的信息可以被提取出来。

5. 层数更深、特征图更宽。使用连续的小卷积核代替大的卷积核，网络的深度更深，并且对边缘进行填充，卷积的过程并不会降低图像尺寸。

6. 全连接转卷积（测试阶段）。在网络测试阶段将训练阶段的三个全连接替换为三个卷积，使得测试得到的全卷积网络因为没有全连接的限制，因而可以接收任意宽或高为的输入。
