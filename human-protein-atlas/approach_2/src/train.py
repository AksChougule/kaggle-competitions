import os
import ast
import torch
import torch.nn as nn
from model_dispatcher import MODEL_DISPATCHER
from datasets import HumanProteinAtlasTrain
from tqdm import tqdm

DEVICE = "cuda"
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")
IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))


EPOCHS = int(os.environ.get("EPOCHS"))

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN"))
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS= ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))
BASE_MODEL = os.environ.get("BASE_MODEL")


def loss_fn(outputs, targets):
    o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15, o16, o17, o18, o19, o20, o21, o22, o23, o24, o25, o26, o27 = outputs
    t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    l4 = nn.CrossEntropyLoss()(o4, t4)
    l5 = nn.CrossEntropyLoss()(o5, t5)
    l6 = nn.CrossEntropyLoss()(o6, t6)
    l7 = nn.CrossEntropyLoss()(o7, t7)
    l8 = nn.CrossEntropyLoss()(o8, t8)
    l9 = nn.CrossEntropyLoss()(o9, t9)
    l10 = nn.CrossEntropyLoss()(o10, t10)
    l11 = nn.CrossEntropyLoss()(o11, t11)
    l12 = nn.CrossEntropyLoss()(o12, t12)
    l13 = nn.CrossEntropyLoss()(o13, t13)
    l14 = nn.CrossEntropyLoss()(o14, t14)
    l15 = nn.CrossEntropyLoss()(o15, t15)
    l16 = nn.CrossEntropyLoss()(o16, t16)
    l17 = nn.CrossEntropyLoss()(o17, t17)
    l18 = nn.CrossEntropyLoss()(o18, t18)
    l19 = nn.CrossEntropyLoss()(o19, t19)
    l20 = nn.CrossEntropyLoss()(o20, t20)
    l21 = nn.CrossEntropyLoss()(o21, t21)
    l22 = nn.CrossEntropyLoss()(o22, t22)
    l23 = nn.CrossEntropyLoss()(o23, t23)
    l24 = nn.CrossEntropyLoss()(o24, t24)
    l25 = nn.CrossEntropyLoss()(o25, t25)
    l26 = nn.CrossEntropyLoss()(o26, t26)
    l27 = nn.CrossEntropyLoss()(o27, t27) 
    
    return (l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9 + l10 + l11 + l12 + l13 + l14 + l15 + l16 + l17 + l18 + l19 + l20 + l21 + l22 + l23 + l24 + l25 + l26 + l27) / 28

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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # some schedulers need to step up after batch and some need steps up after epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.3, verbose=True)
    
    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)
        
    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch} training phase")
        train(train_dataset, train_loader, model, optimizer)
        print(f"Starting epoch {epoch} validation phase")
        val_score = evaluate(valid_dataset, valid_loader, model)
        print(val_score)
        scheduler.step(val_score)
        torch.save(model.state_dict(), f"../models/{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin")
        
if __name__ == "__main__":
    main()
    
