import torchvision
import torch.nn as nn

from models.coatnet import CoAtNet

class CNNClassifier(nn.Module):
    def __init__(self, img_size=224, in_channels=3, num_classes=21843, model_type = 'ResNet18', pretrained = False):
        super(CNNClassifier, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
    
        if model_type == 'ResNet18':
            self.classifier = torchvision.models.resnet18(pretrained)
        elif model_type == 'ResNet50':
            self.classifier = torchvision.models.resnet50(pretrained)
        elif model_type == 'ResNet101':
            self.classifier = torchvision.models.resnet101(pretrained)
        elif model_type == 'AlexNet':
            self.classifier = torchvision.models.alexnet(pretrained)
        elif model_type == 'DenseNet121':
            self.classifier = torchvision.models.densenet121(pretrained)
        elif model_type == 'Vgg16':
            self.classifier = torchvision.models.vgg16(pretrained)
        elif model_type == 'MobileNet_v2':
            self.classifier = torchvision.models.mobilenet_v2(pretrained)
        elif model_type == 'EfficientNet_b6':
            self.classifier = torchvision.models.efficientnet_b6(pretrained)
        elif model_type == 'EfficientNet_b5':
            self.classifier = torchvision.models.efficientnet_b5(pretrained)
        elif 'CoAtNet' in model_type:
            if '0' in model_type:
                num_blocks = [2, 2, 3, 5, 2]            # L
                channels = [64, 96, 192, 384, 768]      # D
                block_types = ['C', 'C', 'T', 'T']
            elif '1' in model_type:
                num_blocks = [2, 2, 6, 14, 2]
                channels = [64, 96, 192, 384, 768]
                block_types = ['C', 'C', 'T', 'T']
            elif '2' in model_type:
                num_blocks = [2, 2, 6, 14, 2]
                channels = [128, 128, 256, 512, 1026]
                block_types = ['C', 'C', 'T', 'T']
            elif '3' in model_type:
                num_blocks = [2, 2, 6, 14, 2]
                channels = [192, 192, 384, 768, 1536]
                block_types = ['C', 'C', 'T', 'T']
            elif '4' in model_type:
                num_blocks = [2, 2, 12, 28, 2]
                channels = [192, 192, 384, 768, 1536]
                block_types = ['C', 'C', 'T', 'T']
            elif '5' in model_type: #changing DIM head from 32 to 64
                num_blocks = [2, 2, 12, 28, 2]
                channels = [192, 256, 512, 1280, 2048]
                block_types = ['C', 'C', 'T', 'T']
            self.classifier = CoAtNet((img_size, img_size), in_channels, num_blocks, channels, num_classes, block_types = block_types)
            
        #change the first conv layer according to the model_type
        if in_channels != 3:
            if 'ResNet' in model_type:
                self.classifier.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = self.classifier.conv1.out_channels, kernel_size = self.classifier.conv1.kernel_size, stride = self.classifier.conv1.stride, padding = self.classifier.conv1.padding, bias = True if not self.classifier.conv1.bias is None else False)
            elif model_type in ['AlexNet', 'Vgg16']:
                self.classifier.features[0]= nn.Conv2d(in_channels = in_channels, out_channels = self.classifier.features[0].out_channels, kernel_size = self.classifier.features[0].kernel_size, stride = self.classifier.features[0].stride, padding = self.classifier.features[0].padding, bias = True if not self.classifier.features[0].bias is None else False)
            elif 'DenseNet' in model_type:
                self.classifier.features.conv0= nn.Conv2d(in_channels = in_channels, out_channels = self.classifier.features.conv0.out_channels, kernel_size = self.classifier.features.conv0.kernel_size, stride = self.classifier.features.conv0.stride, padding = self.classifier.features.conv0.padding, bias = True if not self.classifier.features.conv0.bias is None else False)
            elif 'MobileNet' in model_type or 'EfficientNet' in model_type:
                self.classifier.features[0][0]= nn.Conv2d(in_channels = in_channels, out_channels = self.classifier.features[0][0].out_channels, kernel_size = self.classifier.features[0][0].kernel_size, stride = self.classifier.features[0][0].stride, padding = self.classifier.features[0][0].padding, bias = True if not self.classifier.features[0][0].bias is None else False)
            elif 'CoAtNet' in model_type:
                self.classifier.s0[0][0] = nn.Conv2d(in_channels = in_channels, out_channels = self.classifier.s0[0][0].out_channels, kernel_size = self.classifier.s0[0][0].kernel_size, stride = self.classifier.s0[0][0].stride, padding = self.classifier.s0[0][0].padding, bias = True if not self.classifier.s0[0][0].bias is None else False)
                
        
        #change the last selected fc layer according to the model_type
        if 'ResNet' in model_type:
            fc = nn.Linear(in_features = self.classifier.fc.in_features, out_features=num_classes)
            if self.classifier.fc.bias is None:
                fc.bias = None
            self.classifier.fc = fc
        elif model_type in ['AlexNet', 'Vgg16']:
            fc = nn.Linear(in_features = self.classifier.classifier[6].in_features, out_features=num_classes)
            if self.classifier.classifier[6].bias is None:
                fc.bias = None
            self.classifier.classifier[6]= fc
        elif 'DenseNet' in model_type:
            fc = nn.Linear(in_features = self.classifier.classifier.in_features, out_features=num_classes)
            if self.classifier.classifier.bias is None:
                fc.bias = None
            self.classifier.classifier= fc
        elif 'MobileNet' in model_type or 'EfficientNet' in model_type:
            fc = nn.Linear(in_features = self.classifier.classifier[1].in_features, out_features=num_classes)
            if self.classifier.classifier[1].bias is None:
                fc.bias = None
            self.classifier.classifier[1] = fc
        elif 'CoAtNet' in model_type:
            fc = nn.Linear(in_features = self.classifier.fc.in_features, out_features=num_classes)
            if self.classifier.fc.bias is None:
                fc.bias = None
            self.classifier.fc = fc
        
    def forward(self, x):
        logits = self.classifier(x)
        
        return logits
