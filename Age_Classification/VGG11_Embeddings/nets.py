import torch
import torch.nn as nn

def vgg_block_single(in_ch, out_ch, kernel_size=5, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
def vgg_block_double(in_ch, out_ch, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        #nn.BatchNorm2d(out_ch), #LayerNorm, GroupNorm
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=2, stride=2)       
        )

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MyVGG11(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=4, stride=4),
            LayerNorm(16, eps=1e-6, data_format="channels_first")
        )

        self.conv_block1 =vgg_block_single(in_ch,4)
        self.conv_block2 =vgg_block_single(4,8)

        self.conv_block3 =vgg_block_double(16,32)
        self.conv_block4 =vgg_block_double(32,32)
        self.conv_block5 =vgg_block_double(32,32) #192
        self.embedding_block = nn.Linear(192, 32)
        self.fc_layers = nn.Sequential(
            #nn.Linear(4608, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        
        #x=self.conv_block1(x)
        #x=self.conv_block2(x)
        x = self.stem(x)
        x=self.conv_block3(x)
        
        x=self.conv_block4(x)
        
        #x=self.conv_block5(x)
        x=x.view(x.size(0), -1)

        x = self.embedding_block(x)
        
        x=self.fc_layers(x)

        return x   #checar si hay que ponerle sigmoid

class ageVGG11(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=4, stride=4),
            LayerNorm(16, eps=1e-6, data_format="channels_first")
        )

        self.conv_block1 =vgg_block_single(in_ch,16)
        self.conv_block2 =vgg_block_single(16,16)

        self.conv_block3 =vgg_block_double(16,32)
        self.conv_block4 =vgg_block_double(32,32)
        self.conv_block5 =vgg_block_double(32,64)
        
        self.embedding_block = nn.Linear(192, 32)
        self.fc_layers = nn.Sequential(
            #nn.Linear(4608, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):

        x = self.stem(x)
        x=self.conv_block3(x)
        
        x=self.conv_block4(x)
        
        #x=self.conv_block5(x)
        x=x.view(x.size(0), -1)

        x = self.embedding_block(x)
        
        x=self.fc_layers(x)

        return x   #checar si hay que ponerle sigmoid

class embeddingVGG11(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=4, stride=4),
            LayerNorm(16, eps=1e-6, data_format="channels_first")
        )

        self.conv_block1 =vgg_block_single(in_ch,16)
        self.conv_block2 =vgg_block_single(16,16)

        self.conv_block3 =vgg_block_double(16,32)
        self.conv_block4 =vgg_block_double(32,32)
        self.conv_block5 =vgg_block_double(32,64)
        
        self.embedding_block = nn.Linear(192, 32)
        self.fc_layers = nn.Sequential(
            nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):

        x = self.stem(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(x.size(0), -1)
        x = self.embedding_block(x)
        x = self.fc_layers(x)

        return x

