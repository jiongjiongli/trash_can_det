import json
from pathlib import Path
import random
import cv2
import pandas as pd
from xml.etree import ElementTree as ET
import logging
import yaml
import numpy as np
import torch
from PIL import Image


def set_logging(log_file_path):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
        handlers=[logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler()])


def set_random_seed(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)



class DataConfigManager:
    def __init__(self, config_file_path_dict):
        self.config_file_path_dict = config_file_path_dict

        self.class_name_dict = {
            0: 'background',
            1: 'dustbin',
            2: 'litter_bin',
            3: 'garbage',
            4: 'other_trash_cans'
        }

        self.class_colors = [
              0,   0,   0, # 0 Black,  background
            255, 255,   0, # 1 Yellow, dustbin
              0, 255,   0, # 2 Green,  litter_bin
            255,   0,   0, # 3 Red,    garbage
              0,   0, 255, # 4 Blue,   other_trash_cans
        ]

    def generate(self):
        anno_info_list = self.parse_anno_info()
        self.generate_yolo_configs(anno_info_list)


    def parse_anno_info(self):
        anno_info_list = []
        ignored_image_file_paths = []
        ignored_mask_file_paths = []

        data_root_path = self.config_file_path_dict['path']
        anno_file_paths = list(data_root_path.rglob('*.xml'))

        for anno_file_path in anno_file_paths:
            xml_tree = ET.parse(anno_file_path.as_posix())
            root = xml_tree.getroot()

            filename = root.find('filename').text
            image_file_path = anno_file_path.parent / filename

            if not image_file_path.exists():
                # log_text = r'Warning: image_file_path {} not exist!'.format(image_file_path)
                # logging.warning(log_text)
                ignored_image_file_paths.append(image_file_path)
                continue

            mask_file_path = image_file_path.with_suffix('.png')

            if not mask_file_path.exists():
                ignored_mask_file_paths.append(mask_file_path)
                continue

            anno_info = {
                'image_file_path': image_file_path,
                'mask_file_path': mask_file_path
            }

            anno_info_list.append(anno_info)

        valid_num_info = r'Valid num_files: {} ignored num_files: {} ignored num_mask_files: {}'.format(
            len(anno_info_list),
            len(ignored_image_file_paths),
            len(ignored_mask_file_paths))
        logging.info(valid_num_info)

        return anno_info_list

    def generate_yolo_configs(self,
                              anno_info_list,
                              max_num_val_data=1000,
                              max_val_percent=0.2,
                              seed=7,
                              is_mask_visual=False):
        config_file_path_dict = self.config_file_path_dict
        target_classes = {}

        for class_index, class_name in self.class_name_dict.items():
            if class_index == 0:
                continue

            target_class_index = class_index - 1
            target_classes[target_class_index] = class_name

        for anno_info in anno_info_list:
            mask_file_path = anno_info['mask_file_path']

            mask_img = cv2.imread(mask_file_path.as_posix(),
                                  cv2.IMREAD_GRAYSCALE)
            image_height = mask_img.shape[0]
            image_width = mask_img.shape[1]
            image_size = np.array([image_width, image_height], dtype=np.float32)
            anno_contents = []

            for class_index in self.class_name_dict.keys():
                if class_index == 0:
                    continue

                target_class_index = class_index - 1

                mask_per_class = np.zeros_like(mask_img)
                mask_per_class[mask_img==class_index] = 1

                contours, _ = cv2.findContours(mask_per_class,
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    normed_contour = contour / image_size
                    polygon_flatten = normed_contour.flatten()

                    line = r'{} {}'.format(
                        target_class_index,
                        ' '.join(map(str, polygon_flatten)))
                    anno_contents.append(line)

            if is_mask_visual:
                # Create new image of correct size with mode 'P', set image data
                mask_visual_img = Image.new('P', (mask_img.shape[1], mask_img.shape[0]), 0)
                mask_visual_img.putdata(mask_img.flatten())

                # Set up and apply palette data
                mask_visual_img.putpalette(self.class_colors)

                # Save image
                mask_visual_file_name = r'{}_mask_visual{}'.format(
                    mask_file_path.stem, mask_file_path.suffix)
                mask_visual_file_path = mask_file_path.parent / mask_visual_file_name
                mask_visual_img.save(mask_visual_file_path.as_posix())

            image_file_path = Path(anno_info['image_file_path'])
            anno_config_file_path = image_file_path.with_suffix('.txt')

            with open(anno_config_file_path, 'w') as file_stream:
                for line in anno_contents:
                    file_stream.write('{}\n'.format(line))

        set_random_seed(seed)
        random.shuffle(anno_info_list)

        num_val_data = min(max_num_val_data,
                           int(len(anno_info_list) * max_val_percent))

        anno_infos_dict = {
        'train': anno_info_list[:-num_val_data],
        'val': anno_info_list[-num_val_data:]
        }

        for data_type, anno_infos in anno_infos_dict.items():
            message = r'{}: writing file {} with num_data {}'.format(
                data_type,
                config_file_path_dict[data_type],
                len(anno_infos))
            logging.info(message)

            with open(config_file_path_dict[data_type], 'w') as file_stream:
                for anno_info in anno_infos:
                    image_file_path = anno_info['image_file_path']
                    file_stream.write('{}\n'.format(image_file_path))

        dataset_config = {
            'path': config_file_path_dict['path'].as_posix(),
            'train': config_file_path_dict['train'].name,
            'val': config_file_path_dict['val'].name,
            'names': target_classes
        }

        message = r'Writing dataset config file: {}'.format(
            config_file_path_dict['dataset'])
        logging.info(message)

        logging.info('dataset_config:')
        logging.info(dataset_config)

        with open(config_file_path_dict['dataset'], 'w') as file_stream:
            yaml.dump(dataset_config, file_stream, indent=4)


def main():
    data_root_path = Path(r'/home/data')

    config_file_path_dict = {
        'path': data_root_path,
        'train': data_root_path / 'seg_train.txt',
        'val': data_root_path / 'seg_val.txt',
        'dataset': data_root_path / 'seg_custom_dataset.yaml'
    }

    log_file_path = '/project/train/log/log.txt'

    set_logging(log_file_path)

    logging.info('=' * 80)
    logging.info('Start DataConfigManager')
    data_manager = DataConfigManager(config_file_path_dict)
    data_manager.generate()
    logging.info('End DataConfigManager')
    logging.info('=' * 80)


if __name__ == '__main__':
    main()
