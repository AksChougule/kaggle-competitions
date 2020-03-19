export CUDA_VISIBLE_DEVICES=0

export EPOCHS=10
export LR='0.0001'
export FROZEN_BODY_TRAINING=0
export FBT_EPOCHS=4
export FBT_LR='0.03'

export TRAIN_BATCH_SIZE=1
export TEST_BATCH_SIZE=2

export IMG_HEIGHT=512
export IMG_WIDTH=512
export MODEL_MEAN="(0.485, 0.456, 0.406)"
export MODEL_STD="(0.229, 0.224, 0.225)"
export BASE_MODEL="resnet50"
export TRAINING_FOLDS_CSV="../input/train_folds.csv"
#export TRAINING_FOLDS_CSV="../input/train_folds_oversampled_v2.csv"
#export TRAINING_FOLDS_CSV="../input/train_folds_combined_data.csv"

export TRAINING_FOLDS="(0, 1, 2, 3)"
export VALIDATION_FOLDS="(4,)"
python train_clr.py

export TRAINING_FOLDS="(0, 1, 4, 3)"
export VALIDATION_FOLDS="(2,)"
python train_clr.py

export TRAINING_FOLDS="(0, 4, 2, 3)"
export VALIDATION_FOLDS="(1,)"
python train_clr.py

export TRAINING_FOLDS="(4, 1, 2, 3)"
export VALIDATION_FOLDS="(0,)"
python train_clr.py

export TRAINING_FOLDS="(0, 1, 2, 4)"
export VALIDATION_FOLDS="(3,)"
python train_clr.py
