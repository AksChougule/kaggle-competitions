The approch 2 is purely based on PyTorch and not fastai. The details of all the experiments below:

|  Score (Pvt / Pub)	| Arch	| Epochs (stage1/stage2)	| lr 	| Stratified samlping	| Cross-validation	| Augmentation	| dlr	| Freeze/Unfreeze	| TTA	| External data	| Comment|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| first working version	| 0.29507 / 0.30990 |	ResNet34 | 5	| 1.00E-04	| No	| No	| No	| No	| No	| No	| No	|
| pre1	| 0.34665 / 0.37241	| ResNet50	| 5	| 1.00E-04	| Yes	| No	| No	| No	| No	| No	| No	| Ran only fold 0 from 5-folds|
| pre2	| 0.317 to 0.358 / 0.315 to 0.356	| ResNet50	| 5	| 1.00E-04	| Yes	| Yes	| No	| No	| No	| No	| No	| Ran all 5-folds|
| pre3	| 0.372 to 0.386 / 0.367 to 0.412	| ResNet50	| 10	| 1.00E-04	| Yes	| Yes	| Yes	| No	| No	| No	| No	| Ran all 5-folds|
| dlr v1	| 0.350 to 0.353 / 0.356 to 0.362	| ResNet50	| 10	| 1.00E-04	| Yes	| Yes	| Yes	| Yes	| Yes	| No	| No	| Ran only 2-folds|
| dlr v2	| 0.30361 / 0.30349	| ResNet50	| 3 / 10	| 3e-2 / 1e-2	| Yes	| Yes	| Yes	| Yes	| Yes	| No	| No	| Ran only 1-folds|
| dlr v3	| 0.375 / 0.4028	| ResNet50	| 3 / 10	| 1e-3 / 1e-4	| Yes	| Yes	| Yes	| Yes	| Yes	| No	| No	| Ran only 4-folds|
| dlr v4	| 0.349 / 0.362	| ResNet50	| 3 / 13	| 3e-2 / 1e-4	| Yes	| Yes	| Yes	| Yes	| Yes	| No	| No	| Ran only 2-folds|
| dlr v5	| 0.379 / 0.394	| ResNet50	| 3 / 13	| 1e-3 / 1e-4	| Yes	| Yes	| Yes	| Yes	| Yes	| No	| No	| Ran only 4-folds|
