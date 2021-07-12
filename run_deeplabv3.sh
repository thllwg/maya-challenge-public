python -m src.models.train_model -d data/processed/ --batch-size 16 --num-workers 0 --model-type deeplabv3 --log-dir runs --oversampling false --val-split 1.0 --advanced-augmentation
