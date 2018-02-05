# 算法综述比较

| 算法名称 | 显卡   | 运行速度 | acc recall fmeasure  | 思路   |
| ---- | ---- | ---- | ---- | ---- |
|      |      |      |      |      |  

 Text-CNN&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ICDAR-2013 0.93    0.73    0.82     Text-CNN(attention text) + CE-MSERs detector   

 based on HED model&nbsp;&nbsp;   ICDAR-2013 0.8888  0.8022  0.8433    HED+FCN+VGGmodel+segmentation  //识别斜体  

CTPN          0.14s/per    &nbsp;&nbsp;ICDAR-2013 0.93    0.83    0.88 &nbsp;&nbsp;&nbsp;Side-refinement+blstm+vgg+Fine-scale proposals+bounding box  

Text-boxes                                0.09s/per &nbsp;&nbsp;ICDAR3end2end 0.91    0.89    0.84     &nbsp;&nbsp; SSD +CRNN  

SegLink                       50ms/per ICDAR2013  85.3(f)       &nbsp;&nbsp;            多方向box，SSD+segment+box的角度信息  

R2CNN                                   K80   &nbsp;0.39/per&nbsp;     ICDAR2015  63.60%  61.24%  62.40%     &nbsp;&nbsp;faster-rcnn+three Pooled size  

EAST        &nbsp;                            TITANX&nbsp; 16.8(FPS)&nbsp; ICDAR 2015 0.7833  0.8327  0.8027    &nbsp;&nbsp; PVAnet+fcn+NMS                     