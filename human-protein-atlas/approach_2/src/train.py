import os
import ast
import torch
import torch.nn as nn
from model_dispatcher import MODEL_DISPATCHER
from datasets import HumanProteinAtlasTrain
from model_utils import *
from loss_fns import loss_fn
from tqdm import tqdm
import pdb

DEVICE = "cuda"
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")
IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))

EPOCHS = int(os.environ.get("EPOCHS"))
LR = float(os.environ.get("LR"))
FROZEN_BODY_TRAINING = int(os.environ.get("FROZEN_BODY_TRAINING"))
FBT_EPOCHS = int(os.environ.get("FBT_EPOCHS"))
FBT_LR = float(os.environ.get("FBT_LR"))

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN"))
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS= ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))
BASE_MODEL = os.environ.get("BASE_MODEL")



def train(dataset, data_loader, model, optimizer):
    model.train()
    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
        image = d["image"]
        t0 = d['nucleoplasmn']
        t1 = d['nuclear_membrane']
        t2 = d['nucleoli']
        t3 = d['nucleoli_fibrillar_center']
        t4 = d['nuclear_speckles']
        t5 = d['nuclear_bodies']
        t6 = d['endoplasmic_reticulum']
        t7 = d['golgi_apparatus']
        t8 = d['peroxisomes']
        t9 = d['endosomes']
        t10 = d['lysosomes']
        t11 = d['intermediate_filaments']
        t12 = d['actin_filaments']
        t13 = d['focal_adhesion_sites']
        t14 = d['microtubules']
        t15 = d['microtubule_ends']
        t16 = d['cytokinetic_bridge']
        t17 = d['mitotic_spindle']
        t18 = d['microtubule_organizing_center']
        t19 = d['centrosome']
        t20 = d['lipid_droplets']
        t21 = d['plasma_membrane']
        t22 = d['cell_junctions']
        t23 = d['mitochondria']
        t24 = d['aggresome']
        t25 = d['cytosol']
        t26 = d['cytoplasmic_bodies']
        t27 = d['rods_rings']
        
        image = image.to(DEVICE, dtype=torch.float)
        t0 = t0.to(DEVICE, dtype=torch.long)
        t1 = t1.to(DEVICE, dtype=torch.long)
        t2 = t2.to(DEVICE, dtype=torch.long)
        t3 = t3.to(DEVICE, dtype=torch.long)
        t4 = t4.to(DEVICE, dtype=torch.long)
        t5 = t5.to(DEVICE, dtype=torch.long)
        t6 = t6.to(DEVICE, dtype=torch.long)
        t7 = t7.to(DEVICE, dtype=torch.long)
        t8 = t8.to(DEVICE, dtype=torch.long)
        t9 = t9.to(DEVICE, dtype=torch.long)
        t10 = t10.to(DEVICE, dtype=torch.long)
        t11 = t11.to(DEVICE, dtype=torch.long)
        t12 = t12.to(DEVICE, dtype=torch.long)
        t13 = t13.to(DEVICE, dtype=torch.long)
        t14 = t14.to(DEVICE, dtype=torch.long)
        t15 = t15.to(DEVICE, dtype=torch.long)
        t16 = t16.to(DEVICE, dtype=torch.long)
        t17 = t17.to(DEVICE, dtype=torch.long)
        t18 = t18.to(DEVICE, dtype=torch.long)
        t19 = t19.to(DEVICE, dtype=torch.long)
        t20 = t20.to(DEVICE, dtype=torch.long)
        t21 = t21.to(DEVICE, dtype=torch.long)
        t22 = t22.to(DEVICE, dtype=torch.long)
        t23 = t23.to(DEVICE, dtype=torch.long)
        t24 = t24.to(DEVICE, dtype=torch.long)
        t25 = t25.to(DEVICE, dtype=torch.long)
        t26 = t26.to(DEVICE, dtype=torch.long)
        t27 = t27.to(DEVICE, dtype=torch.long) 
       
        optimizer.zero_grad()
        outputs = model(image)
        targets = (t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27)
        loss = loss_fn(outputs, targets)
        
        loss.backward()
        optimizer.step()
                     

def evaluate(dataset, data_loader, model):
    model.eval()
    final_loss = 0
    counter = 0
    # without no_grad we quickly run out of memory
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
            counter = counter + 1
            image = d["image"]
            t0 = d['nucleoplasmn']
            t1 = d['nuclear_membrane']
            t2 = d['nucleoli']
            t3 = d['nucleoli_fibrillar_center']
            t4 = d['nuclear_speckles']
            t5 = d['nuclear_bodies']
            t6 = d['endoplasmic_reticulum']
            t7 = d['golgi_apparatus']
            t8 = d['peroxisomes']
            t9 = d['endosomes']
            t10 = d['lysosomes']
            t11 = d['intermediate_filaments']
            t12 = d['actin_filaments']
            t13 = d['focal_adhesion_sites']
            t14 = d['microtubules']
            t15 = d['microtubule_ends']
            t16 = d['cytokinetic_bridge']
            t17 = d['mitotic_spindle']
            t18 = d['microtubule_organizing_center']
            t19 = d['centrosome']
            t20 = d['lipid_droplets']
            t21 = d['plasma_membrane']
            t22 = d['cell_junctions']
            t23 = d['mitochondria']
            t24 = d['aggresome']
            t25 = d['cytosol']
            t26 = d['cytoplasmic_bodies']
            t27 = d['rods_rings'] 
    
            image = image.to(DEVICE, dtype=torch.float)
            t0 = t0.to(DEVICE, dtype=torch.long)
            t1 = t1.to(DEVICE, dtype=torch.long)
            t2 = t2.to(DEVICE, dtype=torch.long)
            t3 = t3.to(DEVICE, dtype=torch.long)
            t4 = t4.to(DEVICE, dtype=torch.long)
            t5 = t5.to(DEVICE, dtype=torch.long)
            t6 = t6.to(DEVICE, dtype=torch.long)
            t7 = t7.to(DEVICE, dtype=torch.long)
            t8 = t8.to(DEVICE, dtype=torch.long)
            t9 = t9.to(DEVICE, dtype=torch.long)
            t10 = t10.to(DEVICE, dtype=torch.long)
            t11 = t11.to(DEVICE, dtype=torch.long)
            t12 = t12.to(DEVICE, dtype=torch.long)
            t13 = t13.to(DEVICE, dtype=torch.long)
            t14 = t14.to(DEVICE, dtype=torch.long)
            t15 = t15.to(DEVICE, dtype=torch.long)
            t16 = t16.to(DEVICE, dtype=torch.long)
            t17 = t17.to(DEVICE, dtype=torch.long)
            t18 = t18.to(DEVICE, dtype=torch.long)
            t19 = t19.to(DEVICE, dtype=torch.long)
            t20 = t20.to(DEVICE, dtype=torch.long)
            t21 = t21.to(DEVICE, dtype=torch.long)
            t22 = t22.to(DEVICE, dtype=torch.long)
            t23 = t23.to(DEVICE, dtype=torch.long)
            t24 = t24.to(DEVICE, dtype=torch.long)
            t25 = t25.to(DEVICE, dtype=torch.long)
            t26 = t26.to(DEVICE, dtype=torch.long)
            t27 = t27.to(DEVICE, dtype=torch.long) 

            outputs = model(image)
            targets = (t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27)
            loss = loss_fn(outputs, targets)
            final_loss += loss
    return final_loss /counter
                 


def main():    
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    #model = model_.model
    model.to(DEVICE)
    
    train_dataset = HumanProteinAtlasTrain(
        folds=TRAINING_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
        )
    
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4
        )
    
    valid_dataset = HumanProteinAtlasTrain(
        folds=VALIDATION_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
        )
    
    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4
        )
    
    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)
    
    stage_multiplier = 2
    if FROZEN_BODY_TRAINING==0:
        FBR_EPOCH = FBT_LR = 0

    # We are going to do stage-1 and stage-2 multiple times (stage_multiplier times)
    for i in range(1,stage_multiplier):
        print(f'Stage multiplier: {i}, EPOCHS: {EPOCHS}, LR: {LR}, FBT_EPOCHS: {FBT_EPOCHS}, FBT_LR: {FBT_LR}')
        total_epochs = i*(FBT_EPOCHS + EPOCHS)
          
        ## STAGE 1
       
        if FROZEN_BODY_TRAINING:
            # freeze all layers except the last one
            freeze(model)
            freeze_till_last(model)
            #print(f'EPOCHS: {EPOCHS}, LR: {LR}, FBT_EPOCHS: {FBT_EPOCHS}, FBT_LR: {FBT_LR}')
            lr=FBT_LR
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.3, verbose=True)
        
            for epoch in range(FBT_EPOCHS):
                print(f"Starting epoch {epoch} training phase")
                train(train_dataset, train_loader, model, optimizer)
                print(f"Starting epoch {epoch} validation phase")
                val_score = evaluate(valid_dataset, valid_loader, model)
                print(val_score)
                scheduler.step(val_score)
    
        ## STAGE 2
    
        # unfreeze all layers
        unfreeze(model)
        lr = LR
        # For differential learning rate let's divide the network into 3 groups
        param_groups = [
            [model.model.conv1, model.model.bn1, model.model.layer1],
            [model.model.layer2, model.model.layer3, model.model.layer4],
            [model.model.last_linear,model.l0,model.l1,model.l2,model.l3,model.l4,model.l5,model.l6,model.l7,model.l8,model.l9,model.l10,model.l11,model.l12,model.l13,model.l14,model.l15,model.l16,model.l17,model.l18,model.l19,model.l20,model.l21,model.l22,model.l23,model.l24,model.l25,model.l26,model.l27]
        ]
        # and define 3 lrs for the 3 groups
        lrs = np.array([3e-5, 1e-4 , lr/5])
        # commenting the following line will disable the dlr
        #optimizer = torch.optim.Adam(get_group_params(param_groups, lrs), lr=lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.3, verbose=True)
    
    
        for epoch in range(EPOCHS):
            print(f"Starting epoch {epoch} training phase")
            train(train_dataset, train_loader, model, optimizer)
            print(f"Starting epoch {epoch} validation phase")
            val_score = evaluate(valid_dataset, valid_loader, model)
            print(val_score)
            scheduler.step(val_score)
        
        torch.save(model.state_dict(), f"../models/temp/{BASE_MODEL}_stage-mutli{i}_total-epochs-done{total_epochs}_fold{VALIDATION_FOLDS[0]}.bin")
        
        if i == (stage_multiplier - 1):
            torch.save(model.state_dict(), f"../models/{BASE_MODEL}_w_tfms_stage-mutli{stage_multiplier}_fbtepochs{FBT_EPOCHS}_epoch{EPOCHS}_fold{VALIDATION_FOLDS[0]}.bin")
        
if __name__ == "__main__":
    main()
    
