## How to use models
1. Copy models to tensorflow/models/research folder
2. Generate tfrecord file:
```
python convert_yaml_to_tf_record.py --output_tfrecord models/ssd_4/train.record --output_label_map models/ssd_4/label_map.pbtxt --config_file data/bosch/train.yaml
python convert_yaml_to_tf_record.py --output_tfrecord models/ssd_4/test.record --output_label_map models/ssd_4/label_map.pbtxt --relative_data_path rgb/test --config_file data/bosch/test.yaml
```
Provided that you copied convert_yaml_to_tf_record.py to `tensorflow/models/research` folder and placed bosch dataset to `tensorflow/models/research/data/bosch` folder

## How to evaluate model
Use
```
python object_detection/eval.py  --pipeline_config_path=models/ssd_4/ssd.config --checkpoint_dir models/ssd_4/train/ --eval_dir=models/ssd_4/eval --run_once
```
If you don't have enough GPU memory, try to use
```
CUDA_VISIBLE_DEVICES="" python object_detection/eval.py  --pipeline_config_path=models/ssd_4/ssd.config --checkpoint_dir models/ssd_4/train/ --eval_dir=models/ssd_4/eval --run_once
```

## How to run tensorboard
```
tensorboard --logdir models/ssd_4
```

## How to export trained model for inference
```
# From tensorflow/models/research/
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${TRAIN_PATH} \
    --output_directory output_inference_graph.pb
```
