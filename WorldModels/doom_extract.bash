for i in `seq 1 64`;
do
  echo worker $i
  CUDA_VISIBLE_DEVICES=-1 python extract.py -c configs/doom.config &
  sleep 1.0
done
