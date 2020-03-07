## Approach 2 is based on plain PyTorch code, no Fastai. 

The intention is to get better at implementing the low level code which offer more flexibility.

### Changes implemeneted so far compared to approach 1:
- Multilabel stratified sampling
- Creating custom model using pretrained architecture
- Mutilmodal model head (1 for each class, so 28 in total)
- More modular and scalable code
- A 5 fold cross-validation (the average doesn't help much but a few individual models perform pretty good)

### To try:
- More augmentation
- Progressive resizing during training
- External data
- Freeze/unfreeze of initial layers
- TTA, mixup, fp16, 
