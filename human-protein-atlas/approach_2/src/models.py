import torch.nn as nn
from torch.nn import functional as F
import pretrainedmodels

"""

This is how the last layer of the original ResNet34 looks like:
    
    (avgpool): AvgPool2d(kernel_size=7, stride=7, padding=0)
    (fc): None
    (last_linear): Linear(in_features=512, out_features=1000, bias=True)

So we will add 512*num_of_possible_class_values for each head. That is 512*2.
And we plan to have 28 heads, one for each class, 
so for each of them we add: self.ln = nn.Linear(512, 2)

"""

class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)
        
        self.l0 = nn.Linear(512, 2)  # nucleoplasmn
        self.l1 = nn.Linear(512, 2)  # nuclear_membrane
        self.l2 = nn.Linear(512, 2)  # nucleoli
        self.l3 = nn.Linear(512, 2)  # nucleoli_fibrillar_center
        self.l4 = nn.Linear(512, 2)  # nuclear_speckles
        self.l5 = nn.Linear(512, 2)  # nuclear_bodies
        self.l6 = nn.Linear(512, 2)  # endoplasmic_reticulum
        self.l7 = nn.Linear(512, 2)  # golgi_apparatus
        self.l8 = nn.Linear(512, 2)  # peroxisomes
        self.l9 = nn.Linear(512, 2)  # endosomes
        self.l10 = nn.Linear(512, 2)  # lysosomes
        self.l11 = nn.Linear(512, 2)  # intermediate_filaments
        self.l12 = nn.Linear(512, 2)  # actin_filaments
        self.l13 = nn.Linear(512, 2)  # focal_adhesion_sites
        self.l14 = nn.Linear(512, 2)  # microtubules
        self.l15 = nn.Linear(512, 2)  # microtubule_ends
        self.l16 = nn.Linear(512, 2)  # cytokinetic_bridge
        self.l17 = nn.Linear(512, 2)  # mitotic_spindle
        self.l18 = nn.Linear(512, 2)  # microtubule_organizing_center
        self.l19 = nn.Linear(512, 2)  # centrosome
        self.l20 = nn.Linear(512, 2)  # lipid_droplets
        self.l21 = nn.Linear(512, 2)  # plasma_membrane
        self.l22 = nn.Linear(512, 2)  # cell_junctions
        self.l23 = nn.Linear(512, 2)  # mitochondria
        self.l24 = nn.Linear(512, 2)  # aggresome
        self.l25 = nn.Linear(512, 2)  # cytosol
        self.l26 = nn.Linear(512, 2)  # cytoplasmic_bodies
        self.l27 = nn.Linear(512, 2)  # rods_rings
        
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        # print(x.shape) # The dim is: bs * 512 * 16 * 16 for input images of 3*512*512
        # Adaptive pooling supports all image sizes
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        # print(x.shape) # The dim is bs * 512
        l0  = self.l0(x)
        l1  = self.l1(x)
        l2  = self.l2(x)
        l3  = self.l3(x)
        l4  = self.l4(x)
        l5  = self.l5(x)
        l6  = self.l6(x)
        l7  = self.l7(x)
        l8  = self.l8(x)
        l9  = self.l9(x)
        l10  = self.l10(x)
        l11  = self.l11(x)
        l12  = self.l12(x)
        l13  = self.l13(x)
        l14  = self.l14(x)
        l15  = self.l15(x)
        l16  = self.l16(x)
        l17  = self.l17(x)
        l18  = self.l18(x)
        l19  = self.l19(x)
        l20  = self.l20(x)
        l21  = self.l21(x)
        l22  = self.l22(x)
        l23  = self.l23(x)
        l24  = self.l24(x)
        l25  = self.l25(x)
        l26  = self.l26(x)
        l27  = self.l27(x)

        return l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27
    

"""

This is how the last layer of the original ResNet50 looks like:
    
    (avgpool): AvgPool2d(kernel_size=7, stride=7, padding=0)
    (fc): None
    (last_linear): Linear(in_features=2048, out_features=1000, bias=True)

So we will add 2048*num_of_possible_class_values for each head. That is 2048*2.
And we plan to have 28 heads, one for each class, 
so for each of them we add: self.ln = nn.Linear(2048, 1)

"""

class ResNet50(nn.Module):
    def __init__(self, pretrained):
        super(ResNet50, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet50"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet50"](pretrained=None)
        
        self.l0 = nn.Linear(2048, 2)  # nucleoplasmn
        self.l1 = nn.Linear(2048, 2)  # nuclear_membrane
        self.l2 = nn.Linear(2048, 2)  # nucleoli
        self.l3 = nn.Linear(2048, 2)  # nucleoli_fibrillar_center
        self.l4 = nn.Linear(2048, 2)  # nuclear_speckles
        self.l5 = nn.Linear(2048, 2)  # nuclear_bodies
        self.l6 = nn.Linear(2048, 2)  # endoplasmic_reticulum
        self.l7 = nn.Linear(2048, 2)  # golgi_apparatus
        self.l8 = nn.Linear(2048, 2)  # peroxisomes
        self.l9 = nn.Linear(2048, 2)  # endosomes
        self.l10 = nn.Linear(2048, 2)  # lysosomes
        self.l11 = nn.Linear(2048, 2)  # intermediate_filaments
        self.l12 = nn.Linear(2048, 2)  # actin_filaments
        self.l13 = nn.Linear(2048, 2)  # focal_adhesion_sites
        self.l14 = nn.Linear(2048, 2)  # microtubules
        self.l15 = nn.Linear(2048, 2)  # microtubule_ends
        self.l16 = nn.Linear(2048, 2)  # cytokinetic_bridge
        self.l17 = nn.Linear(2048, 2)  # mitotic_spindle
        self.l18 = nn.Linear(2048, 2)  # microtubule_organizing_center
        self.l19 = nn.Linear(2048, 2)  # centrosome
        self.l20 = nn.Linear(2048, 2)  # lipid_droplets
        self.l21 = nn.Linear(2048, 2)  # plasma_membrane
        self.l22 = nn.Linear(2048, 2)  # cell_junctions
        self.l23 = nn.Linear(2048, 2)  # mitochondria
        self.l24 = nn.Linear(2048, 2)  # aggresome
        self.l25 = nn.Linear(2048, 2)  # cytosol
        self.l26 = nn.Linear(2048, 2)  # cytoplasmic_bodies
        self.l27 = nn.Linear(2048, 2)  # rods_rings
        
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        # print(x.shape) # The dim is: bs * 2048 * 16 * 16 for images of size 3*512*512
        # Adaptive pooling supports all image sizes
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        # print(x.shape) # The dim is: bs * 2048
        l0  = self.l0(x)
        l1  = self.l1(x)
        l2  = self.l2(x)
        l3  = self.l3(x)
        l4  = self.l4(x)
        l5  = self.l5(x)
        l6  = self.l6(x)
        l7  = self.l7(x)
        l8  = self.l8(x)
        l9  = self.l9(x)
        l10  = self.l10(x)
        l11  = self.l11(x)
        l12  = self.l12(x)
        l13  = self.l13(x)
        l14  = self.l14(x)
        l15  = self.l15(x)
        l16  = self.l16(x)
        l17  = self.l17(x)
        l18  = self.l18(x)
        l19  = self.l19(x)
        l20  = self.l20(x)
        l21  = self.l21(x)
        l22  = self.l22(x)
        l23  = self.l23(x)
        l24  = self.l24(x)
        l25  = self.l25(x)
        l26  = self.l26(x)
        l27  = self.l27(x)

        return l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27


class ResNet50_v2(nn.Module):
    def __init__(self, pretrained):
        super(ResNet50_v2, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet50"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet50"](pretrained=None)
        
        self.classes = nn.Linear(2048, 28)  
        
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        # print(x.shape) # The dim is: bs * 2048 * 16 * 16 for images of size 3*512*512
        # Adaptive pooling supports all image sizes
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        # print(x.shape) # The dim is: bs * 2048
        classes  = self.classes(x)


        return classes

class ResNet50_v3(nn.Module):
    def __init__(self, pretrained):
        super(ResNet50_v3, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet50"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet50"](pretrained=None)
        
        self.l0 = nn.Linear(2048, 1)  # nucleoplasmn
        self.l1 = nn.Linear(2048, 1)  # nuclear_membrane
        self.l2 = nn.Linear(2048, 1)  # nucleoli
        self.l3 = nn.Linear(2048, 1)  # nucleoli_fibrillar_center
        self.l4 = nn.Linear(2048, 1)  # nuclear_speckles
        self.l5 = nn.Linear(2048, 1)  # nuclear_bodies
        self.l6 = nn.Linear(2048, 1)  # endoplasmic_reticulum
        self.l7 = nn.Linear(2048, 1)  # golgi_apparatus
        self.l8 = nn.Linear(2048, 1)  # peroxisomes
        self.l9 = nn.Linear(2048, 1)  # endosomes
        self.l10 = nn.Linear(2048, 1)  # lysosomes
        self.l11 = nn.Linear(2048, 1)  # intermediate_filaments
        self.l12 = nn.Linear(2048, 1)  # actin_filaments
        self.l13 = nn.Linear(2048, 1)  # focal_adhesion_sites
        self.l14 = nn.Linear(2048, 1)  # microtubules
        self.l15 = nn.Linear(2048, 1)  # microtubule_ends
        self.l16 = nn.Linear(2048, 1)  # cytokinetic_bridge
        self.l17 = nn.Linear(2048, 1)  # mitotic_spindle
        self.l18 = nn.Linear(2048, 1)  # microtubule_organizing_center
        self.l19 = nn.Linear(2048, 1)  # centrosome
        self.l20 = nn.Linear(2048, 1)  # lipid_droplets
        self.l21 = nn.Linear(2048, 1)  # plasma_membrane
        self.l22 = nn.Linear(2048, 1)  # cell_junctions
        self.l23 = nn.Linear(2048, 1)  # mitochondria
        self.l24 = nn.Linear(2048, 1)  # aggresome
        self.l25 = nn.Linear(2048, 1)  # cytosol
        self.l26 = nn.Linear(2048, 1)  # cytoplasmic_bodies
        self.l27 = nn.Linear(2048, 1)  # rods_rings
        
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        # print(x.shape) # The dim is: bs * 2048 * 16 * 16 for images of size 3*512*512
        # Adaptive pooling supports all image sizes
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        # print(x.shape) # The dim is: bs * 2048
        l0  = self.l0(x)
        l1  = self.l1(x)
        l2  = self.l2(x)
        l3  = self.l3(x)
        l4  = self.l4(x)
        l5  = self.l5(x)
        l6  = self.l6(x)
        l7  = self.l7(x)
        l8  = self.l8(x)
        l9  = self.l9(x)
        l10  = self.l10(x)
        l11  = self.l11(x)
        l12  = self.l12(x)
        l13  = self.l13(x)
        l14  = self.l14(x)
        l15  = self.l15(x)
        l16  = self.l16(x)
        l17  = self.l17(x)
        l18  = self.l18(x)
        l19  = self.l19(x)
        l20  = self.l20(x)
        l21  = self.l21(x)
        l22  = self.l22(x)
        l23  = self.l23(x)
        l24  = self.l24(x)
        l25  = self.l25(x)
        l26  = self.l26(x)
        l27  = self.l27(x)

        return l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27
