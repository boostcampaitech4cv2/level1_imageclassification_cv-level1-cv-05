import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import timm

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.lin1= nn.Linear(128,3)
        self.lin2= nn.Linear(128,2)
        self.lin3= nn.Linear(128,3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        o1=self.lin1(x)
        o2=self.lin2(x)
        o3=self.lin3(x)
        return o1,o2,o3


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.res = models.resnet50(pretrained=True)
        
        self.res = timm.create_model('vit_large_patch16_224', pretrained = True)
        # model.head = nn.Linear(in_features = 1024, out_features = num_classes)
    
        # self.freeze()
        self.mask_model = nn.Sequential(nn.Dropout(0.3),nn.Linear(1024,128),nn.BatchNorm1d(128),nn.Softplus(beta = 2),
                                        nn.Dropout(0.5),nn.Linear(128,3))
        self.gen_model = nn.Sequential(nn.Dropout(0.3),nn.Linear(1024,128),nn.BatchNorm1d(128),nn.Softplus(beta = 2),
                                        nn.Dropout(0.5),nn.Linear(128,1))
        self.age_model = nn.Sequential(nn.Dropout(0.3),nn.Linear(1024,128),nn.BatchNorm1d(128),nn.Softplus(beta = 2),
                                        nn.Dropout(0.5),nn.Linear(128,3))
        self.sig = nn.Sigmoid()
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
    def freeze(self, a = False):
        for i in self.res.parameters():
            i.requires_grad = a
    
        

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.res(x)
        out1 =self.mask_model(x)
        out2 =self.sig(self.gen_model(x)).view(-1)
        out3 =self.age_model(x)
        # print(out2)
        # print(out2.type())
        return out1, out2, out3
