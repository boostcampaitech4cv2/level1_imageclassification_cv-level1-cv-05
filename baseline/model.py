import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch

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
    def __init__(self):
        super().__init__()
        self.res = models.resnet50(pretrained=True)
        
        self.mask_model = nn.Sequential(nn.Softplus(beta = 2),nn.Dropout(0.3),nn.Linear(1000,128),nn.BatchNorm1d(128),nn.Softplus(beta = 2),
                                        nn.Dropout(0.5),nn.Linear(128,3))
        self.gen_model = nn.Sequential(nn.Softplus(beta = 2),nn.Dropout(0.3),nn.Linear(1000,128),nn.BatchNorm1d(128),nn.Softplus(beta = 2),
                                        nn.Dropout(0.5),nn.Linear(128,1))
        self.age_mask_model= nn.Sequential(nn.Softplus(beta = 2),nn.Dropout(0.3),nn.Linear(1000,128),nn.Softplus(beta = 2),
                                        nn.Dropout(0.5),nn.Linear(128,3))
        self.age_no_mask_model= nn.Sequential(nn.Softplus(beta = 2),nn.Dropout(0.3),nn.Linear(1000,128),nn.Softplus(beta = 2),
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
    def forward(self, x, y = None):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        if not self.training : #eval
            x = self.res(x)
            out1 = self.mask_model(x)
            out2 =self.sig(self.gen_model(x)).view(-1)
            mask_out = torch.argmax(out1, dim=-1)
            out3 = torch.ones(mask_out.shape[0],3).to(out2.device)
            mask_off = x[mask_out == 2]
            mask_on = x[mask_out != 2]
            
            out3[mask_out == 2] = self.age_no_mask_model(mask_off)
            out3[mask_out != 2] = self.age_mask_model(mask_on)
            return out1, out2, out3
        else:
            x = self.res(x)
            out1 = self.mask_model(x)
            out2 =self.sig(self.gen_model(x)).view(-1)
            mask_out = torch.argmax(out1, dim=-1)
            out3 = torch.ones(mask_out.shape[0],3).to(out2.device)
            mask_off = x[mask_out == 2]
            mask_on = x[mask_out != 2]
            
            out3[mask_out == 2] = self.age_no_mask_model(mask_off)
            out3[mask_out != 2] = self.age_mask_model(mask_on)
            # x = self.res(x)
            # out1 = self.mask_model(x)
            # out2 =self.sig(self.gen_model(x)).view(-1)
            # mask_GT = x[y == 2]
            # no_mask_GT = x[y != 2]
            
            # out3 = torch.ones(out1.shape[0],3).to(out2.device)
            # out3[y==2] = self.age_no_mask_model(mask_GT)
            # out3[y!=2] = self.age_mask_model(no_mask_GT)
            return out1, out2, out3
# class MyModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = models.resnet34(pretrained=True)
#         self.mask_model = nn.Linear(1000,3)
#         self.gen_model = nn.Linear(1000,1)
#         self.age_mask = nn.Linear(1000,3)
#         self.age_noramal = nn.Linear(1000,3)