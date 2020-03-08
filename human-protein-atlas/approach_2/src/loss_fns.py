#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 19:50:36 2020

@author: Akshay

This is a collection of loss functions

"""
import torch.nn as nn

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
