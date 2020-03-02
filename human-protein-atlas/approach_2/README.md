Approach 2 is based on only PyTorch code, no Fastai. 

The intention is to learn to implement the low lower code which offer more flexibility.

A few changes implemeneted so far compared to approach 1:
- A 5 fold cross-validation
- Multilabel stratified sampling
- More modular and scalable code
- Mutilmodal model head (1 for each class, so 28 in total)
