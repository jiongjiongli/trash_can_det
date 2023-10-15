
cd /project/train/src_repo/ultralytics
export PYTHONPATH=$PYTHONPATH:/project/train/src_repo/ultralytics

echo 'Reset env...'
mkdir -p /project/train/models

rm -rf /project/train/tensorboard/*
mkdir -p /project/train/tensorboard

rm -rf /project/train/log/*
mkdir -p /project/train/log

rm -rf /project/train/result-graphs/*
mkdir -p /project/train/result-graphs

# echo 'Start data_config...'
# python trash_can_det/data_config.py

# echo 'Start cvmart_train...'
# python trash_can_det/yolo_train.py

echo 'Start seg_data_config...'
python trash_can_det/seg_data_config.py

echo 'Start seg_yolo_train...'
python trash_can_det/seg_yolo_train.py

echo 'Completed!'
