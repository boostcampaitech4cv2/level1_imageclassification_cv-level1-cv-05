import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import timm

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

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
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self,batch_size=32,age_cls=4):
        super().__init__()
        self.res = timm.create_model('vit_base_patch16_224', pretrained = True)
        
        # self.res = models.resnet152(pretrained=True)
        self.res.head=nn.Linear(768,256)
        # self.freeze(True)
    
        self.batch_size=batch_size
        self.age_cls=age_cls
        self.mask_model = nn.Sequential(nn.BatchNorm1d(256),nn.Softplus(beta = 2),nn.Dropout(0.5),
                                        nn.Linear(256,64),nn.BatchNorm1d(64),nn.Softplus(beta = 2),nn.Dropout(0.5),
                                        nn.Linear(64,3))
        self.gen_model = nn.Sequential(nn.BatchNorm1d(256),nn.Softplus(beta = 2),nn.Linear(256,64),nn.BatchNorm1d(64),
                                       nn.Softplus(beta = 2),nn.Dropout(0.5),nn.Linear(64,1))
        self.age_model = nn.Sequential(nn.BatchNorm1d(256),nn.Softplus(beta = 2),nn.Dropout(0.5),
                                        nn.Linear(256,64),nn.BatchNorm1d(64),nn.Softplus(beta = 2),nn.Dropout(0.5),
                                        nn.Linear(64,age_cls))
        self.sig = nn.Sigmoid()
        # self.init_param()
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
    def init_param(self):
        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.kaiming_normal_(i.weight)
                nn.init.zeros_(i.bias)
    def freeze(self, a = False):
        for i in self.res.parameters():
            i.requires_grad =False if a else True
        for i in self.res.head.parameters():
            i.requires_grad = True
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.res(x)
        out1 =self.mask_model(x)
        out2 =self.sig(self.gen_model(x)).view(-1)
        out3 =self.age_model(x)

        return out1, out2, out3
class MyModel2(nn.Module):
    def __init__(self,batch_size = 32, age_cls = 5):
        super().__init__()
        self.res = timm.create_model('vit_base_patch16_224', pretrained = True)
        # self.res = models.resnet152(pretrained=True)
        self.res.head=nn.Linear(768,256)
        self.batch_size = batch_size
        self.age_cls = age_cls
        # self.freeze(True)
        self.mask_model = nn.Sequential(nn.BatchNorm1d(256),nn.Softplus(beta = 2),nn.Dropout(0.5),
                                        nn.Linear(256,64),nn.BatchNorm1d(64),nn.Softplus(beta = 2),nn.Dropout(0.5),
                                        nn.Linear(64,3))
        self.gen_model = nn.Sequential(nn.BatchNorm1d(256),nn.Softplus(beta = 2),nn.Linear(256,64),nn.BatchNorm1d(64),
                                    nn.Softplus(beta = 2),nn.Dropout(0.5),nn.Linear(64,1))
        self.age_mask_model = nn.Sequential(nn.Softplus(beta = 2),nn.Dropout(0.5),nn.Linear(256,256),
                                            nn.Softplus(beta = 2),nn.Dropout(0.5),nn.Linear(256,64),
                                            nn.Softplus(beta = 2),nn.Dropout(0.5),nn.Linear(64,5))
        self.age_no_mask_model = nn.Sequential(nn.Softplus(beta = 2),nn.Dropout(0.5),nn.Linear(256,256),
                                            nn.Softplus(beta = 2),nn.Dropout(0.5),nn.Linear(256,64),
                                            nn.Softplus(beta = 2),nn.Dropout(0.5),nn.Linear(64,5))
        self.sig = nn.Sigmoid()
        # self.init_param()
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
    def freeze(self, a = False):
        for i in self.res.parameters():
            i.requires_grad =False if a else True
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.res(x)
        out1 = self.mask_model(x)
        out2 = self.sig(self.gen_model(x)).view(-1)
        out3 = torch.ones(out2.shape[0],self.age_cls).to(out1.device)
        pred = torch.argmax(out1, dim=-1)
        out3[pred==2] = self.age_no_mask_model(x[pred==2])
        out3[pred!=2] = self.age_mask_model(x[pred!=2])
        return out1, out2, out3
    def init_param(self):
            for i in self.res.head.modules():
                if isinstance(i, nn.Linear):
                    nn.init.kaiming_normal_(i.weight)
                    nn.init.zeros_(i.bias)
            for i in self.mask_model.modules():
                if isinstance(i, nn.Linear):
                    nn.init.kaiming_normal_(i.weight)
                    nn.init.zeros_(i.bias)
            for i in self.gen_model.modules():
                if isinstance(i, nn.Linear):
                    nn.init.kaiming_normal_(i.weight)
                    nn.init.zeros_(i.bias)
            for i in self.age_mask_model.modules():
                if isinstance(i, nn.Linear):
                    nn.init.kaiming_normal_(i.weight)
                    nn.init.zeros_(i.bias)
            for i in self.age_no_mask_model.modules():
                if isinstance(i, nn.Linear):
                    nn.init.kaiming_normal_(i.weight)
                    nn.init.zeros_(i.bias)