

def main():
    repo_dir_path = Path('/project/train/src_repo')
    model_file_path = repo_dir_path / 'yolov8n-seg.pt'
    data_root_path = Path(r'/home/data')
    dataset_config_file_path = data_root_path / 'seg_custom_dataset.yaml'
    model_save_dir_path = Path('/project/train/models')
    # model_file_path = model_save_dir_path  / 'train/weights' / 'last.pt'
    result_graphs_dir_path = Path('/project/train/result-graphs')
    font_file_names = ['Arial.ttf']
    log_file_path = Path('/project/train/log/log.txt')

    file_handler = logging.FileHandler(log_file_path.as_posix(), mode='a')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    tb_logger = TensorboardLogger()
    tb_callbacks['on_pretrain_routine_start'] = tb_logger.on_pretrain_routine_start
    tb_callbacks['on_train_start'] = tb_logger.on_train_start
    tb_callbacks['on_fit_epoch_end'] = tb_logger.on_fit_epoch_end
    tb_callbacks['on_batch_end'] = tb_logger.on_batch_end

    result_graphs_dir_path.mkdir(parents=True, exist_ok=True)

    for font_file_name in font_file_names:
        font_file_path = repo_dir_path / font_file_name
        dest_file_path = USER_CONFIG_DIR / font_file_name
        shutil.copyfile(font_file_path, dest_file_path)

    model = YOLO(model_file_path.as_posix())
    results = model.train(
        data=dataset_config_file_path.as_posix(),
        batch=8,
        seed=7,
        epochs=150,
        project=model_save_dir_path.as_posix())

if __name__ == '__main__':
    main()
