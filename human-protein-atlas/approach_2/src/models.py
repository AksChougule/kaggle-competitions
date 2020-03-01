import torch.nn as nn
from torch.nn import functional as F
import pretrainedmodels

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
        # Adaptive pooling supports all image sizes
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
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