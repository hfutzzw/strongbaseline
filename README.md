
**由于Github上传文件大小限制，我们团队将初赛的训练模型文件（包括权重参数）、预训练模型权重文件 存储在百度网盘**
链接如下：
[初赛训练模型文件](https://pan.baidu.com/s/18VBTsjO31pHOcnbQVe1fSQ),
[预训练模型文件](https://pan.baidu.com/s/1-2QoE_MKvJAb-6EzD0-YVQ)
使用时请将初赛训练模型文件放置在./logs文件夹下，将预训练模型文件放置在./pretrained_models文件夹下

# 项目运行办法


## 项目的文件结构

![Image Name](https://cdn.kesci.com/upload/image/q1u8bpjvq1.png?imageView2/0/w/960/h/960)

``` 
                config     默认实验参数配置
                configs    特定实验参数配置
		data/datasets 数据集接口文件(naic.py为本次比赛的数据集接口）
		data/samplers  平衡采样器实现
		data/transforms 数据增强文件
		data/eval_reid.py  Rank1和Map测试指标实现文件
		engine   训练和测试引擎，简化训练测试重复代码
		layers/triplet_loss.py  损失函数
		logs  训练日志及训练模型保存
		modeling/backbones  预训练模型实现
		modeling/baseline.py strongbasline模型
		modeling/Pyramid.py  最终比赛方案模型
		pretrained_models/*  预训练模型权重文件
		solver/ 优化器、学习率调节器、
		tools/train.py  训练流程定义
		tools/test.py 测试流程定义
		utils/re_ranking.py reranking实现文件
```

## 项目的运行步骤
首先需要确保数据集路径配置正确，数据集路径格式如下

```
NAIC/trainset
      /train_set
      /train_list.txt
	
NAIC/testset
      /query_a
      /gallery_a
      /query_a_list.txt
	
NAIC/testsetB
      /query_b
      /gallery_b

```
		     
训练阶段：
Step1. 修改实验参数配置 ```configs/resnet50_s_t_c_NAIC_Pyramid.yml```
Step2. ```python tools/train.py --config_file='configs/resnet50_s_t_c_NAIC.yml' MODEL.DEVICE_ID "('8,9')"```


在运行测试代码前，首先需要修改```./data/dataset/naic.py```中第46行为```use_split_testset = False```,修改```./data/datasets/eval_reid.py```第26行为```submit = True```。

测试阶段：
``` python
python tools/test.py --config_file='./configs/resnet50_s_t_c_NAIC_Pyramid.yml' MODEL.DEVICE_ID "('2')" TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')"  TEST.RE_RANKING ('yes')" TEST.WEIGHT "('./logs/sf_tri_center_NAIC_res50_ibn_a_pyramid_woREA_trainall/resnet50_ibn_a_model_360.pth')"
```

## 运行结果的位置

1. 训练权重路径：
```./logs/sf_tri_center_NAIC_res50_ibn_a_pyramid_woREA_trainall/resnet50_ibn_a_model_360.pth```
2. 训练日志路径：
```./logs/sf_tri_center_NAIC_res50_ibn_a_pyramid_woREA_trainall/log.txt```
3. 提交结果路径：
```./submission_a.json，./submission_b.json```

**由于Github上传文件大小限制，我们团队将初赛训练模型文件（包括权重参数）、预训练模型权重文件 存储在百度网盘**
[初赛训练模型文件](https://pan.baidu.com/s/18VBTsjO31pHOcnbQVe1fSQ)
[预训练模型文件](https://pan.baidu.com/s/1-2QoE_MKvJAb-6EzD0-YVQ)
在使用时，分别将初赛训练模型文件放置到```./logs```文件夹下，将预训练模型文件放置到```./pretrained_models```文件夹下
