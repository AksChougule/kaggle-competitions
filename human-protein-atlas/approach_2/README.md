## Approach 2 is based on plain PyTorch code, no Fastai. 

The intention is to get better at implementing the low level code which offer more flexibility.

### Changes implemeneted so far compared to approach 1:
- Multilabel stratified sampling
- Creating custom model using pretrained architecture
- Mutilmodal model head (1 for each class, so 28 in total)
- More augmentation (HorizontalFlip,RandomBrightness, RandomContrast, RandomRotate90 and ShiftScaleRotate helps)
- A 5 fold cross-validation (the average doesn't help much but a few individual models perform pretty good)
- Freeze/unfreeze of initial layers using pytorch (has not helped yet, maybe need to play around lr a lit).
- More modular and scalable code

### To try:
- Progressive resizing during training
- External data
- TTA, mixup, fp16, 
