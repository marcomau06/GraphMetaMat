from src.dataset import DataLoaderFactory
from src.generative_curve.model import ModelEnsemble, Model
from src.generative_curve.pretrain import pretrain_curves, pretrain_property
from src.generative_curve.pretrain_graphs import pretrain_graphs
from src.generative_curve.train import train, TrainConfig
from src.generative_curve.test import test, plot_pred
from src.config import args
from src.utils import write_to_log, write_plot_loss, log_dir, get_optimizer
from pprint import pformat

import os
import torch
import time
import pickle

DEVICE = args['device']

def main():
    ######## Seed for reproducible results ##########
    import random
    import numpy as np
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    #################################################
    
    t0 = time.time()
    # save_cfg()
    
    ############# Dataset #############################
    dlf = DataLoaderFactory(**args['dataset'])
    dataset_contrastive = dlf.get_train_dataset_contrastive() if args['forward']['train_config']['use_contrastive'] else None
    dataset_train = dlf.get_train_dataset()
    dataset_valid = dlf.get_valid_dataset()
    dataset_test = dlf.get_test_dataset()
   
    ############### Model initialization ###############
    model = Model.init_from_cfg(dataset=dataset_train, **args['forward_model'])
    print('Learnable parameters',sum(p.numel() for p in model.parameters()))
    print(model)

    ############### Load pretrained model ###############
    if 'load_model' in args and args['load_model'] != 'None' and args['load_model'] is not None:
        state_dict = torch.load(args['load_model'], map_location=torch.device(DEVICE))
        for z in [x for x in state_dict if 'decoder' in x or 'encoder_rho' in x]:
            state_dict.pop(z)
        model.load_state_dict(state_dict, strict=False)
        model.encoder.add_adapters()

    ############## Optimizer init. ##################
    optimizer = get_optimizer(model=model, **args['optimizer'])

    ############## Training ##################
    # best_iter = 0
    # loss_train = 0.0
    best_iter, loss_train =\
       train(
           dataset_train, dataset_valid, dataset_contrastive, model, optimizer,
           TrainConfig(**args['forward']['train_config']), args['dataset']['batch_size'],
           device=DEVICE
       )

    ############### Load best model ###############
    write_to_log(log_dir, f'best_iter: {best_iter}\n')
    if args['forward']['train_config']['use_snapshot'] is not None:
        model = ModelEnsemble.from_path(log_dir, model)
        # for i in range(len(model.model_li)):
        #     model.model_li[i].decoder_rag.reset_datastore(
        #         dataset_train, model.model_li[i].encoder, model.model_li[i].pooler)
    else:
        model.load_state_dict(torch.load(os.path.join(log_dir, f'model_best.pt')))
        # model.decoder_rag.reset_datastore(dataset_train, model.encoder, model.pooler)

    ############### Test ###############
    metrics_collated, metrics_raw, graph_li, curve_pred_li, curve_true_li = \
        test(
            dataset_test, model,
            device=DEVICE,
            plot_num_samples_max=args['forward']['train_config']['plot_num_samples_max'])
        
    ############### Plotting and logging ###############
    plot_dir = os.path.join(log_dir, 'plots_test')
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    plot_pred(graph_li, curve_pred_li, curve_true_li,
              skip_magnitude=args['dataset']['curve_norm_cfg']['curve_method'] is None,
              dn=plot_dir, max_figs=args['forward']['train_config']['plot_num_samples_max'])

    write_to_log(log_dir, pformat(metrics_collated))
    write_to_log(log_dir, f'runtime: {time.time()-t0}s')
    write_plot_loss(log_dir, loss_train, args['forward']['train_config']['num_epochs'], plot_suffix = 'train')

    ############### Export results ###############
    if 'export_results' in args and args['export_results']:
        export_obj = \
            {
                'metrics_collated':metrics_collated, 'metrics_raw':metrics_raw,
                'graph_li':graph_li, 'curve_pred_li':curve_pred_li,
                'curve_true_li':curve_true_li
            }
        with open(os.path.join(log_dir, f'results.pkl'), "wb") as f:
            pickle.dump(export_obj, f)

if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(False):
        main()


