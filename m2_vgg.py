import torch
import os
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms


 
#超参数设置
#DEVICE=torch.device('cuda'if torch.cuda.is_available() else 'cpu')          #转gpu
DEVICE=torch.device( 'cpu')          #转cpu
print(DEVICE)
EPOCH=2
BATCH_SIZE=256
 
#网络模型构建VGG-16
#注意：连接层参数与原版参数不一样,因为输入为65x65,注释内为输入为65情况下各层输出大小的计算
#另外，预测时未按照原论文将张量过一下softmax，后续有空加入
#本人因为显存不够，无法完整训练，仅仅确保了能够训练
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
 
#数据预处理
 
#数据路径
# aim_dir0=r'shdj'
# aim_dir1=r'sdj'
# source_path0=r'efeeeeee'
# source_path1=r'ddddddddd'
 
#数据增强
# def DataEnhance(sourth_path,aim_dir,size):
#     name=0
#     #得到源文件的文件夹
#     file_list=os.listdir(sourth_path)
#     #创建目标文件的文件夹
#     if not os.path.exists(aim_dir):
#         os.mkdir(aim_dir)
#
#     for i in file_list:
#         img=Image.open('%s\%s'%(sourth_path,i))
#         print(img.size)
#
#         name+=1
#         transform1=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.ToPILImage(),
#             transforms.Resize(size),
#         ])
#         img1=transform1(img)
#         img1.save('%s/%s'%(aim_dir,name))
#
#         name+=1
#         transform2=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.ToPILImage(),
#             transforms.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5)
#         ])
#         img2 = transform1(img)
#         img2.save('%s/%s' % (aim_dir, name))
#
#         name+=1
#         transform3=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.ToPILImage(),
#             transforms.RandomCrop(227,pad_if_needed=True),
#             transforms.Resize(size)
#         ])
#         img3 = transform1(img)
#         img3.save('%s/%s' % (aim_dir, name))
#
#         name+=1
#         transform4=transforms.Compose([
#             transforms.Compose(),
#             transforms.ToPILImage(),
#             transforms.RandomRotation(60),
#             transforms.Resize(size),
#         ])
#         img4 = transform1(img)
#         img4.save('%s/%s' % (aim_dir, name))
#
#
# DataEnhance(source_path0,aim_dir0,size)
# DataEnhance(source_path1,aim_dir1,size)
 
#对文件区分为训练集，测试集，验证集
 
#归一化处理
normalize=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
 
#训练集
path_1=r'.\train_0'
trans_1=transforms.Compose([
    transforms.Resize((65,65)),
    transforms.ToTensor(),
    normalize,
])
 
#数据集
train_set=ImageFolder(root=path_1,transform=trans_1)
#数据加载器
train_loader=torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,
                                         shuffle=True,num_workers=0)
print(train_set.classes)
 
#测试集
path_2=r'.\train_0'
trans_2=transforms.Compose([
    transforms.Resize((65,65)),
    transforms.ToTensor(),
    normalize,
])
test_data=ImageFolder(root=path_2,transform=trans_2)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE,
                                        shuffle=True,num_workers=0)
 
#验证集
path_3=r'.\train_0'
valid_data=ImageFolder(root=path_2,transform=trans_2)
valid_loader=torch.utils.data.DataLoader(valid_data,batch_size=BATCH_SIZE,
                                         shuffle=True,num_workers=0)
 
#定义模型
model=VGG().to(DEVICE)
#优化器的选择
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0005)
 
 
#训练过程
def train_model(model,device,train_loader,optimizer,epoch):
    train_loss=0
    model.train()
    for batch_index,(data,label) in enumerate(train_loader):
        data,label=data.to(device),label.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.cross_entropy(output,label)
        loss.backward()
        optimizer.step()
        if batch_index%300==0:
            train_loss=loss.item()
            print('Train Epoch:{}\ttrain loss:{:.6f}'.format(epoch,loss.item()))
 
    return  train_loss
 
 
#测试部分的函数
def test_model(model,device,test_loader):
    model.eval()
    correct=0.0
    test_loss=0.0
 
    #不需要梯度的记录
    with torch.no_grad():
        for data,label in test_loader:
            data,label=data.to(device),label.to(device)
            output=model(data)
            test_loss+=F.cross_entropy(output,label).item()
            pred=output.argmax(dim=1)
            correct+=pred.eq(label.view_as(pred)).sum().item()
        test_loss/=len(test_loader.dataset)
        print('Test_average_loss:{:.4f},Accuracy:{:3f}\n'.format(
            test_loss,100*correct/len(test_loader.dataset)
        ))
        acc=100*correct/len(test_loader.dataset)
 
        return test_loss,acc
 
 
#训练开始
list=[]
Train_Loss_list=[]
Valid_Loss_list=[]
Valid_Accuracy_list=[]
 
#Epoc的调用
for epoch in range(1,EPOCH+1):
    #训练集训练
    train_loss=train_model(model,DEVICE,train_loader,optimizer,epoch)
    Train_Loss_list.append(train_loss)
    torch.save(model,r'.\model%s.pth'%epoch)
 
    #验证集进行验证
    test_loss,acc=test_model(model,DEVICE,valid_loader)
    Valid_Loss_list.append(test_loss)
    Valid_Accuracy_list.append(acc)
    list.append(test_loss)
 
#验证集的test_loss
 
min_num=min(list)
min_index=list.index(min_num)
 
print('model%s'%(min_index+1))
print('验证集最高准确率： ')
print('{}'.format(Valid_Accuracy_list[min_index]))
 
#取最好的进入测试集进行测试
model=torch.load(r'.\model%s.pth'%(min_index+1))
model.eval()
 
accuracy=test_model(model,DEVICE,test_loader)
print('测试集准确率')
print('{}%'.format(accuracy))
 
 
#绘图
#字体设置，字符显示
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
 
#坐标轴变量含义
x1=range(0,EPOCH)
y1=Train_Loss_list
y2=Valid_Loss_list
y3=Valid_Accuracy_list
 
#图表位置
plt.subplot(221)
#线条
plt.plot(x1,y1,'-o')
#坐标轴批注
plt.ylabel('训练集损失')
plt.xlabel('轮数')
 
plt.subplot(222)
plt.plot(x1,y2,'-o')
plt.ylabel('验证集损失')
plt.xlabel('轮数')
 
plt.subplot(212)
plt.plot(x1,y3,'-o')
plt.ylabel('验证集准确率')
plt.xlabel('轮数')
 
#显示
plt.show()
 
