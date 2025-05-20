import logging
from pathlib import Path

def set_logger(name: str, log_file: str):
    # Get workspace root directory
    project_root = Path('/Users/xiaolong/prediction_competition_2023')
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / Path(log_file).name
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(pathname)s [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger