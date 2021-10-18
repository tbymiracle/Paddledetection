# get paddle weight
# cd ../
#python -u tools/train.py -c configs/detectors/cascade_rcnn_r50_rfp_1x_coco.yml
# cd gfl_debug_tools/
# get torch weight
#wget https://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_rfp_1x_coco/cascade_rcnn_r50_rfp_1x_coco-8cf51bfd.pth
# get the weight name in torch and paddle
#python3.7 match_weight_name.py cascade_rcnn_r50_rfp_1x_coco-8cf51bfd.pth model_final.pdparams torch_paddle_match.txt
# convert the torch weight to paddle
python3.7 convert.py ./cascade_rcnn_r50_rfp_1x_coco-8cf51bfd.pth ./detectors_weight_match/torch_state_dict.txt ./detectors_weight_match/paddle_state_dict.txt ./result/detectors_paddle.pdparams


