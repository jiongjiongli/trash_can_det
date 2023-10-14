# CVMart trash can detect
Trash can detection and garbage segmentation.

# 1. Prepare CVMart Env
```bash
!rm /project/train/src_repo/cvmart_env.sh

!wget -P /project/train/src_repo/ https://raw.githubusercontent.com/jiongjiongli/trash_can_det/main/trash_can_det/cvmart_env.sh

!bash /project/train/src_repo/cvmart_env.sh
```



# 2. Train
```bash
!bash /project/train/src_repo/start_train.sh
```
