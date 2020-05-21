for i in `seq 1 64`;
do
  echo worker $i
  CUDA_VISIBLE_DEVICES=-1 xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python extract.py -c configs/carracing.config &
  sleep 1.0
done
wait
CUDA_VISIBLE_DEVICES=0 python vae_train.py -c configs/carracing.config
CUDA_VISIBLE_DEVICES=0 python series.py -c configs/carracing.config
CUDA_VISIBLE_DEVICES=0 python rnn_train.py -c configs/carracing.config
CUDA_VISIBLE_DEVICES=-1 xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train.py -c configs/carracing.config
