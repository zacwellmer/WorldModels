CUDA_VISIBLE_DEVICES=0 python vae_train.py -c configs/doom.config
CUDA_VISIBLE_DEVICES=0 python series.py -c configs/doom.config
CUDA_VISIBLE_DEVICES=0 python rnn_train.py -c configs/doom.config
CUDA_VISIBLE_DEVICES=-1 python train.py -c configs/doom.config
