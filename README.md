# Adversarial Example Detection with Critical Data Routing Paths

## Requirements
pytorch == 0.3.1
python == 3.5
sklearn == 0.22

## Data Preparation
- prepare imagenet dataset following the instructions in https://github.com/pytorch/examples/tree/master/imagenet, which results in an imagenet folder with train and val sub-folders.
- generate image index by
```bash
python prepare_images_list.py --data_dir IMAGENET_DATA_DIR/train --dump_path data/train_images_list.pkl
python prepare_images_list.py --data_dir IMAGENET_DATA_DIR/val --dump_path data/val_images_list.pkl
```

## Adversarial Example Detection
```bash
python adversarial_detect.py --data IMAGENET_DATA_DIR -a ARCH --gpu GPU_ID
```
where `ARCH` denotes the attacking network (AlexNet, VGG16), `GPU_ID` is the available gpu device number. For ResNet50, run the command
```bash
python adversarial_detect_resnet.py --data IMAGENET_DATA_DIR -a resnet50 --gpu GPU_ID
```
Current setting is one training sample and one testing sample from each class to extract the CDRP used for adversarial example detection. You can adjust the sample number from each class by
```bash
python adversarial_detect.py --data IMAGENET_DATA_DIR -a ARCH --train_num_per_class 5 --test_num_per_class 1 --gpu GPU_ID
```

- Note: we have improved the codes after CVPR paper is published, and current settings can achieve 0.9+ AUROC value.

## Citation
```bibtex
@inproceedings{wang2018cdrp,
	title={Interpret Neural Networks by Identifying Critical Data Routing Paths},
	author={Wang, Yulong and Su, Hang and Zhang, Bo and Hu, Xiaolin},
	booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	pages={8906-8914},
	year={2018},
	publisher = {IEEE},
	address={Salt Lake City, USA}
}
```
