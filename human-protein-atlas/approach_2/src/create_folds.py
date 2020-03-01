import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("/home/ubuntu/datasets/bengali-ai/train.csv")
    print(df.head())
    df.loc[:,"kfold"] = -1
    
    # random shuffle with fraction = 1 then reset the index
    df = df.sample(frac=1).reset_index(drop=True)
    
    X = df.image_id.values
    y = df[["grapheme_root","vowel_diacritic","consonant_diacritic"]].values
    
    # define how many folds
    mskf = MultilabelStratifiedKFold(n_splits=5)
    
    # Let's go through the iterator and get trai and al indices
    for fold, (trn_, val_) in enumerate(mskf.split(X, y)):
        print("TRAIN: ", trn_, "VALID : ",val_)
        df.loc[val_, "kfold"] = fold
        
    print(df.kfold.value_counts())
    df.to_csv("../input/train_folds.csv", index=False)