import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import logging
# import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import H5DataLoader 
from fuxictr.preprocess import FeatureProcessor, build_dataset
import models as model_zoo
import gc
import argparse
import os
from pathlib import Path
from fuxictr.pytorch.models import MultiTaskModel, BaseModel



if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())
    
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        # Build feature_map and transform h5 data
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))
    if 'task_labels' in params:
        if isinstance(params['task_labels'],list):
            task_labels = params['task_labels']
        else: 
            task_labels = list(params['task_labels'])
        feature_map.labels = task_labels
        feature_map.set_column_index()

    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model

    train_gen, valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)

    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()
    
    logging.info('******** Test evaluation ********')
    test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()
    test_result = {}
    if test_gen:
      test_result = model.evaluate(test_gen)
    
    result_filename = os.path.join('results',Path(args['config']).name.replace(".yaml", "") + '.csv')
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                    ' '.join(sys.argv), experiment_id, params['dataset_id'],
                    "N.A.", print_to_list(valid_result), print_to_list(test_result)))
