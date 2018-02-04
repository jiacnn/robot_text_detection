# 算法综述比较

| 算法名称                                | 显卡   | 运行速度 |         准确率   recall  Fmeasure  | 思路   |

| Text-CNN                               | ---- | ----    |ICDAR-2013 0.93    0.73    0。82     |Text-CNN(attention text) + CE-MSERs detector |
| based on HED model                     |      |         |ICDAR-2013 0.8888  0.8022  0.8433    |HED+FCN+VGGmodel+segmentation  |//识别斜体
|CTPN                                    |      |0.14s/per|ICDAR-2013 0.93    0.83    0.88      |Side-refinement+blstm+vgg+Fine-scale proposals+bounding box
|Text-boxes                              |      |0.09s/per|IC3end2end 0.91    0.89    0.84      |SSD +CRNN
|SegLink                                 |      |50ms/per |ICDAR2013  85.3(f)                   |多方向box，SSD+segment+box的角度信息|
|R2CNN                                   |K80   |0.39     |ICDAR2015  63.60%  61.24%  62.40%    | faster-rcnn+three Pooled size
|EAST                                    |TITANX|16.8(FPS)|ICDAR 2015 0.7833  0.8327  0.8027    | PVAnet+fcn+NMS                     