import os
import ast
import time
import torch
import torch.nn as nn
from model_dispatcher import MODEL_DISPATCHER
from datasets import HumanProteinAtlasTrain
from model_utils import *
from loss_fns import loss_fn, loss_fn_v2
from tqdm import tqdm
import pdb
from torchvision import models
import tqdm as tqdm
from CLR import CLR
from OneCycle import OneCycle

DEVICE = "cuda"
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")
IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))

epoch = int(os.environ.get("EPOCHS"))
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

total = 0
correct = 0

train_loss = 0
test_loss = 0
best_acc = 0
trn_losses = []
trn_accs = []
val_losses = []
val_accs = []


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
    
    def accuracy(output, target, is_test=False):
        global total
        global correct
        batch_size = target.size(0)
        total += batch_size
        
        _, pred = torch.max(output, 1)
        if is_test:
            preds.extend(pred)
        correct += (pred == target).sum()
        return 100 * correct / total

    
    class AvgStats(object):
        def __init__(self):
            self.reset()
            
        def reset(self):
            self.losses =[]
            self.precs =[]
            self.its = []
            
        def append(self, loss, prec, it):
            self.losses.append(loss)
            self.precs.append(prec)
            self.its.append(it)

    train_stats = AvgStats()
    test_stats = AvgStats()
    
    # define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.95, weight_decay=1e-4)
    clr = CLR(optimizer, len(train_loader))
    
    def update_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def update_mom(optimizer, mom):
        for g in optimizer.param_groups:
            g['momentum'] = mom

    def save_checkpoint(model, is_best, filename='data/checkpoint.pth.tar'):
        """Save checkpoint if a new best is achieved"""
        if is_best:
            torch.save(model.state_dict(), filename)  # save checkpoint
        else:
            print ("=> Validation Accuracy did not improve")
    
    
    # from fastai library
    def load_checkpoint(model, filename = 'data/checkpoint.pth.tar'):
        sd = torch.load(filename, map_location=lambda storage, loc: storage)
        names = set(model.state_dict().keys())
        for n in list(sd.keys()): 
            if n not in names and n+'_raw' in names:
                if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
                del sd[n]
        model.load_state_dict(sd)


    filename = 'data/clr.pth.tar'
    save_checkpoint(model, True, filename)

    
    #t = tqdm.tqdm(train_loader, leave=False, total=len(train_loader))
    running_loss = 0.
    avg_beta = 0.98
    
    print(len(train_loader))
    
    model.train()
    gpu = True if torch.cuda.is_available() else False
    
    t = tqdm.tqdm(train_loader, leave=False, total=len(train_loader))
    
# =============================================================================
#     for bi, d in enumerate(t):
#     #for bi, d in enumerate(train_loader):
#         image = d["image"]
#         t0 = d['nucleoplasmn']
#         t1 = d['nuclear_membrane']
#         t2 = d['nucleoli']
#         t3 = d['nucleoli_fibrillar_center']
#         t4 = d['nuclear_speckles']
#         t5 = d['nuclear_bodies']
#         t6 = d['endoplasmic_reticulum']
#         t7 = d['golgi_apparatus']
#         t8 = d['peroxisomes']
#         t9 = d['endosomes']
#         t10 = d['lysosomes']
#         t11 = d['intermediate_filaments']
#         t12 = d['actin_filaments']
#         t13 = d['focal_adhesion_sites']
#         t14 = d['microtubules']
#         t15 = d['microtubule_ends']
#         t16 = d['cytokinetic_bridge']
#         t17 = d['mitotic_spindle']
#         t18 = d['microtubule_organizing_center']
#         t19 = d['centrosome']
#         t20 = d['lipid_droplets']
#         t21 = d['plasma_membrane']
#         t22 = d['cell_junctions']
#         t23 = d['mitochondria']
#         t24 = d['aggresome']
#         t25 = d['cytosol']
#         t26 = d['cytoplasmic_bodies']
#         t27 = d['rods_rings'] 
# 
#         image = image.to(DEVICE, dtype=torch.float)
#         t0 = t0.to(DEVICE, dtype=torch.long)
#         t1 = t1.to(DEVICE, dtype=torch.long)
#         t2 = t2.to(DEVICE, dtype=torch.long)
#         t3 = t3.to(DEVICE, dtype=torch.long)
#         t4 = t4.to(DEVICE, dtype=torch.long)
#         t5 = t5.to(DEVICE, dtype=torch.long)
#         t6 = t6.to(DEVICE, dtype=torch.long)
#         t7 = t7.to(DEVICE, dtype=torch.long)
#         t8 = t8.to(DEVICE, dtype=torch.long)
#         t9 = t9.to(DEVICE, dtype=torch.long)
#         t10 = t10.to(DEVICE, dtype=torch.long)
#         t11 = t11.to(DEVICE, dtype=torch.long)
#         t12 = t12.to(DEVICE, dtype=torch.long)
#         t13 = t13.to(DEVICE, dtype=torch.long)
#         t14 = t14.to(DEVICE, dtype=torch.long)
#         t15 = t15.to(DEVICE, dtype=torch.long)
#         t16 = t16.to(DEVICE, dtype=torch.long)
#         t17 = t17.to(DEVICE, dtype=torch.long)
#         t18 = t18.to(DEVICE, dtype=torch.long)
#         t19 = t19.to(DEVICE, dtype=torch.long)
#         t20 = t20.to(DEVICE, dtype=torch.long)
#         t21 = t21.to(DEVICE, dtype=torch.long)
#         t22 = t22.to(DEVICE, dtype=torch.long)
#         t23 = t23.to(DEVICE, dtype=torch.long)
#         t24 = t24.to(DEVICE, dtype=torch.long)
#         t25 = t25.to(DEVICE, dtype=torch.long)
#         t26 = t26.to(DEVICE, dtype=torch.long)
#         t27 = t27.to(DEVICE, dtype=torch.long) 
#         
#         outputs = model(image)
#         targets = (t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27)
#         loss = loss_fn(outputs, targets)
#             
#         #print(loss.data)
#         running_loss = avg_beta * running_loss + (1-avg_beta) *loss.data
#         smoothed_loss = running_loss / (1 - avg_beta**(bi+1))
#         t.set_postfix(loss=smoothed_loss)
#         
#         lr = clr.calc_lr(smoothed_loss)
#         if lr == -1 :
#             break
#         update_lr(optimizer, lr)   
#         
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         
#     
#     clr.plot() 
# =============================================================================
    
    onecycle = OneCycle(int(len(train_dataset) * epoch /TRAIN_BATCH_SIZE), 0.8, prcnt=(epoch - 82) * 100/epoch, momentum_vals=(0.95, 0.8))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.95, weight_decay=1e-4)
    load_checkpoint(model, filename)

        
    def train(epoch=0, use_cycle = False, model=model):
        model.train()
        
        global best_acc
        global trn_accs, trn_losses

        is_improving = True
        counter = 0
        running_loss = 0.
        avg_beta = 0.98
        
        for bi, d in enumerate(t):
        #for i, (input, target) in enumerate(train_loader):
            bt_start = time.time()
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
            
            if use_cycle:    
                lr, mom = onecycle.calc()
                update_lr(optimizer, lr)
                update_mom(optimizer, mom)
            
            outputs = model(image)
            
            targets = (t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27)
            loss = loss_fn(outputs, targets)
            
            o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15, o16, o17, o18, o19, o20, o21, o22, o23, o24, o25, o26, o27 = outputs
            # This stacks array by adding 27 rows below the first one
            oo = torch.cat((o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15, o16, o17, o18, o19, o20, o21, o22, o23, o24, o25, o26, o27), 0)
            # This creates 1D tensor of length 28 with 1s and 0s, for the entire batch, in our case just 1
            oo2 = torch.argmax(oo,dim=1)
            # get the tt2 tensor on cpu, convert to ndarray, find indices where values is 1
            #pred_classes = np.where(oo2.cpu().numpy() == 1)

            tt = torch.cat((t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27),0)
            res = tt-oo2 # 20 sized 1D tensor
            res_len = res.cpu().numpy().size
            acc =  1 - (res.nonzero().size(0)/res_len)
                
            running_loss = avg_beta * running_loss + (1-avg_beta) *loss.data
            smoothed_loss = running_loss / (1 - avg_beta**(bi+1))
            
            trn_losses.append(smoothed_loss)
                
            # measure accuracy and record loss
            prec = acc
            trn_accs.append(prec)
    
            train_stats.append(smoothed_loss, prec, time.time()-bt_start)
            if prec > best_acc :
                best_acc = prec
                save_checkpoint(model, True)
    
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    preds =[]
          
    def test(model=model):
        model.eval()
        
        global val_accs, val_losses

        running_loss = 0.
        
        with torch.no_grad():
            for bi, d in enumerate(t):
            #for i, (input, target) in enumerate(test_loader):
                bt_start = time.time()
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
    
                running_loss = avg_beta * running_loss + (1-avg_beta) *loss.data
                smoothed_loss = running_loss / (1 - avg_beta**(bi+1))
                
            
                o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15, o16, o17, o18, o19, o20, o21, o22, o23, o24, o25, o26, o27 = outputs
                # This stacks array by adding 27 rows below the first one
                oo = torch.cat((o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15, o16, o17, o18, o19, o20, o21, o22, o23, o24, o25, o26, o27), 0)
                # This creates 1D tensor of length 28 with 1s and 0s, for the entire batch, in our case just 1
                oo2 = torch.argmax(oo,dim=1)
                # get the tt2 tensor on cpu, convert to ndarray, find indices where values is 1
                #pred_classes = np.where(oo2.cpu().numpy() == 1)
    
                tt = torch.cat((t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27),0)
                res = tt-oo2 # 20 sized 1D tensor
                res_len = res.cpu().numpy().size
                acc =  1 - (res.nonzero().size(0)/res_len)
                
           
                # measure accuracy and record loss
                prec = acc
                test_stats.append(loss.data, prec, time.time()-bt_start)
    
                val_losses.append(smoothed_loss)
                val_accs.append(prec)
    
    
    def fit(use_onecycle=False, model=model):
        print("Epoch\tTrn_loss\tVal_loss\tTrn_acc\t\tVal_acc")
        for j in range(epoch):
            train(j, use_onecycle, model)
            test(model)
            print("{}\t{:06.8f}\t{:06.8f}\t{:06.8f}\t{:06.8f}"
                  .format(j+1, trn_losses[-1], val_losses[-1], trn_accs[-1], val_accs[-1]))

    fit(True)
    
    # for train
    ep_losses = []
    for i in range(0, len(train_stats.losses), len(train_loader)):
        if i != 0 :
            ep_losses.append(train_stats.losses[i])
        
    # for val
    ep_lossesv = []
    for i in range(0, len(test_stats.losses), len(test_loader)):
        if(i != 0):
            ep_lossesv.append(test_stats.losses[i])
    
    torch.save(model.state_dict(), f"../models/{BASE_MODEL}_w_tfms_v10_epoch{epoch}_fold{VALIDATION_FOLDS[0]}.bin")
        
if __name__ == "__main__":
    main()
    
