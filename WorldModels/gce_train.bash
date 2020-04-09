CUDA_VISIBLE_DEVICES=-1 xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train.py -n 72 
