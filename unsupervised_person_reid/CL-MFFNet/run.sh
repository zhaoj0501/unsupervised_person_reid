CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -b 256 -a resnet50 -d market1501 --iters 200 --eps 0.45 --num-instances 16 --pooling-type avg --memorybank CMhybrid --epochs 60

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -b 256 -a resnet50 -d msmt17 --iters 200 --eps 0.6 --num-instances 16 --pooling-type avg --memorybank CMhybrid --epochs 60 


CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -b 256 -a resnet_ibn50a -d market1501 --iters 200 --eps 0.45 --num-instances 16 --pooling-type gem --memorybank CMhybrid --epochs 60

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -b 256 -a resnet_ibn50a -d msmt17 --iters 200 --eps 0.6 --num-instances 16 --pooling-type gem --memorybank CMhybrid --epochs 60

# test
CUDA_VISIBLE_DEVICES=0 python examples/test.py -d market1501 --data-dir examples/data/market1501 --pooling-type avg