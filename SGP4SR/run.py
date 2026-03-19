import argparse
from logging import getLogger, FileHandler
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color


from data.dataset import SGPDataset
from collections import OrderedDict

from sgp import SGP

from recbole.trainer import Trainer

def get_logger_filename(logger):
    file_handler = next((handler for handler in logger.handlers if isinstance(handler, FileHandler)), None)
    if file_handler:
        filename = file_handler.baseFilename
        print(f"The log file name is {filename}")
    else:
        raise Exception("No file handler found in logger")
    return filename


def run(dataset, setting='SGP4SR.yaml,run.yaml',log_prefix="", **kwargs):
    setting = setting.split(',')
    config = Config(model=SGP, dataset=dataset, config_file_list=setting, config_dict=kwargs)

    config['log_prefix'] = log_prefix

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = SGPDataset(config)

    logger.info(dataset)

    train_data, valid_data, test_data, co_data, colens, copos= data_preparation(config, dataset)
    model = SGP(config, train_data.dataset, co_data, colens).to(config['device'])
    logger.info(model)
    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    logger_Filename = get_logger_filename(logger)
    logger.info(f"Write log to {logger_Filename}")

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='baby', help='dataset name')
    parser.add_argument('-f', type=bool, default=True)
    parser.add_argument('-setting', type=str, default='SGP4SR.yaml,run.yaml')
    parser.add_argument('-note', type=str, default='')
    args, unparsed = parser.parse_known_args()
    print(args)

    run(args.d, setting=args.setting, fix_enc=args.f, log_prefix=args.note)
