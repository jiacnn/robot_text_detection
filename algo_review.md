# 算法速度比较

## 检测算法综述比较

| 算法名称       | 显卡        | 运行速度      | acc recall fmeasure                | 思路                                       |
| ---------- | --------- | --------- | ---------------------------------- | ---------------------------------------- |
| Text-CNN   |           |           | ICDAR-2013 0.93    0.73    0.82    | Text-CNN(attention text) + CE-MSERs detector |
| CTPN       |           | 0.14s/per | ICDAR-2013 0.93    0.83    0.88    | Side-refinement+blstm+vgg+Fine-scale proposals+bounding box |
| Text-boxes |           | 0.09s/per | ICDAR3end2end 0.91    0.89    0.84 | ssd+textboxes layers                     |
| SegLink    | TITAN x*4 | 50ms/per  | 85.3(f)                            | SSD+LINK+segment                         |
| R2CNN      | K80       | 0.39/per  | ICDAR2015  63.60%  61.24%  62.40%  | faster-rcnn+three Pooled size            |
| EAST       | TITANX    | 16.8(FPS) | ICDAR 2015 0.7833  0.8327  0.8027  | PVAnet+fcn+NMS                           |

## 识别算法综述比较

| 算法名称             | 显卡      | 运行速度      | acc recall fmeasure | 思路                                       |
| ---------------- | ------- | --------- | ------------------- | ---------------------------------------- |
| CRNN             |         | 0.16s/per | SVT 0.8             | CNN+BiLSTM+CTC                           |
| SEE              |         |           | FSNS 0.78           | ResNet location <br />CNN+BiLSTM+FC recognition |
| Google Attention | TITAN X |           | FSNS 0.8379         | ssd+textboxes layers                     |

 