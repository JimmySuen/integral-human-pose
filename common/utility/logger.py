import os
import logging
import time

from .folder import make_folder

def create_logger(cfg_file, image_set_list, output_path, log_path):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    
    curDir = os.path.dirname(cfg_file).split('/')[-1]
    cfg_name = os.path.basename(cfg_file).rsplit('.', 1)[0]
    
    # model path
    final_output_path = os.path.join(output_path, curDir, cfg_name, '{}'.format('_'.join(image_set_list)))
    make_folder(final_output_path)

    # log path
    final_log_path = os.path.join(log_path, curDir, cfg_name)
    make_folder(final_log_path)
    
    # create logger 
    logging.basicConfig(filename=os.path.join(final_log_path, '{}_{}.log'.format(cfg_name, time_str)),
                        format='%(asctime)-15s %(message)s', level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    return final_output_path, final_log_path, logger