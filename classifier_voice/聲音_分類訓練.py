file_name='Voice_202201010000'
import torch
device=torch.device('cpu') # 'cuda'/'cpu'，import torch
num_classes=3
batch_size=2
learning_rate=0.0001
step_size=1000 # Reriod of learning rate decay
epochs=100
TrainingVoice='C:\Dropbox\Voice/'

# 建立dataset 
import os
all_voice_name=os.listdir(TrainingVoice) # all_voice_name=['crun','walk']，import os
V=list()
L=list()
import numpy
for i in range(0,len(all_voice_name)):
    files=os.listdir(TrainingVoice+all_voice_name[i]) # files=['R1.npy','R2.npy',...]，import os
    for j in range(0,len(files)):
        V_=torch.tensor(numpy.load(os.path.join(TrainingVoice,all_voice_name[i],files[j]))) # [20,40]
        V_=V_.unsqueeze(0) # [1,20,40]
        V.append(V_)
        L.append(i)
V=torch.stack(V,0) # [10,1,20,40]
L=torch.tensor(L) # [10]

from torch import nn
class Simple_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=2,padding=1), # [2,32,21,41]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # [2,32,10,20]
        )
        self.fc=nn.Sequential(
            nn.Linear(32*10*20,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,num_classes),
        )
    def forward(self,x):
        x=self.block_1(x) 
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

classifier=Simple_Model().to(device)
criterion=nn.CrossEntropyLoss() # 分類
optimizer=torch.optim.Adam(classifier.parameters(),lr=learning_rate) # import torch
#optimizer=torch.optim.SGD(classifier.parameters(),lr=learning_rate) # import torch
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size,0.1) # import torch
train_acc_his,train_losses_his=[],[]
for i in range(1,epochs+1):
    print('Running Epoch:'+str(i))
    p=numpy.random.permutation(len(V)) # 隨機排序
    V=V[[p]]
    L=L[p]
    train_correct,train_loss,train_total=0,0,0
    classifier.train()
    for j in range(0,len(V),batch_size):
        voice=V[[numpy.arange(j,j+batch_size)]] # [batch_size,1,20,40]
        cls=L[numpy.arange(j,j+batch_size)] # [batch_size]
        voice,cls=voice.to(device),cls.to(device)
        pred=classifier(voice) # pred：[batch_size,num_classes]
        loss=criterion(pred,cls) # loss.item()：一個batch的平均loss，[1]
        output_id=torch.max(pred,dim=1)[1] # output_id：網路輸出編號(0表示預測為第一個輸出)，[batch_size]
        train_correct+=numpy.sum(torch.eq(cls,output_id).cpu().numpy()) # 累加計算每一epoch正確預測總數，import numpy
        train_loss+=loss.item()*voice.size(0) # 累加計算每一epoch的loss總和。loss.item()：一個batch的平均loss，[1]。img.size(0)：一個batch的訓練資料總數
        train_total+=voice.size(0) # 累加計算訓練資料總數
        optimizer.zero_grad() # 權重梯度歸零
        loss.backward() # 計算每個權重的loss梯度
        optimizer.step() # 權重更新    
    scheduler.step()

    train_acc=train_correct/train_total*100 # 計算每一個epoch的平均訓練正確率(%)
    train_loss=train_loss/train_total # 計算每一個epoch的平均訓練loss
    train_acc_his.append(train_acc) # 累積紀錄每一個epoch的平均訓練正確率(%)，[epochs]
    train_losses_his.append(train_loss) # 累積記錄每一個epoch的平均訓練loss，[epochs]
    print('Training Loss='+str(train_loss))
    print('Training Accuracy(%)='+str(train_acc))

import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.subplot(211)
plt.plot(train_acc_his,'b',label='trainingaccuracy')
plt.title('Accuracy(%)')
plt.legend(loc='best')
plt.subplot(212)
plt.plot(train_losses_his,'b',label='training loss')
plt.title('Loss')
plt.legend(loc='best')
plt.show()

torch.save(classifier.state_dict(),file_name)