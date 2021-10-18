cd ../
# get paddle loss, run two times
CUDA_VISIBLE_DEVICES=0 python3.7 -u tools/train.py -c configs/gfl/gfl_r50_fpn_1x_coco.yml
# get paddle eval result
CUDA_VISIBLE_DEVICES=0 python3.7 -u tools/eval.py -c configs/gfl/gfl_r50_fpn_1x_coco.yml
