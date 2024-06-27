import matplotlib.pyplot as plt

# from src.utils import log_dir, writer
from src.utils import log_dir
from src.utils import plot, to_numpy

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os
from tqdm import tqdm
from copy import deepcopy
from pprint import pprint

from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

def lt_space(dataset, model, decoder_config, device):
    model.eval()
    model = model.to(device)
    ae_loss = \
        AELoss(
            model.decoder.dim_hidden, decoder_config,
            model.decoder.decoder, model.loss_name,
            model.use_accum, device)
    y_max_pred = []
    y_max_true = []
    latent = []
    for i, curve in enumerate(tqdm(dataset)):        
        with torch.no_grad():
            curve.to(device)
            loss, curve_pred, latent_i = ae_loss(curve, curve.curve_stats)
            if ae_loss.use_accum:
                curve_pred[:, :, 1] = torch.cumsum(curve_pred[:, :, 1], dim=1)
        
        # Store latent vectors for each batch
        latent.append(latent_i)                
        
        curve = model.unnormalize(curve, curve_stats)
        curve_pred = model.unnormalize(curve_pred, curve_stats)
         
        y_max_pred.extend(curve_pred[:,:,1].max(dim=-1)[0].detach().cpu().tolist())
        y_max_true.extend(curve[:,:,1].max(dim=-1)[0].detach().cpu().tolist()) 
        
        E_pred.extend(torch.diff(curve_pred[:,:,1])[0].detach().cpu().tolist()) 
        E_true.extend(torch.diff(curve[:,:,1])[0].detach().cpu().tolist()) 
    
    # Latent space
    latent = torch.cat(latent, dim = 0) # (curves, d)
    tsne_plot(latent.cpu().detach().numpy(), y_max_true, y_max_pred, num_iters)             
    tsne_plot(latent.cpu().detach().numpy(), E_true, E_pred, num_iters)         
    

def pretrain_curves(dataset_pretrain_train, dataset_pretrain_valid, model, optimizer, train_cfg, decoder_config, device):
    model.train()
    model = model.to(device)
    ae_loss = \
        AELoss(
            model.decoder.dim_hidden, decoder_config,
            model.decoder.decoder, model.loss_name,
            model.use_accum, device)

    best_iter = -1
    num_iters = 0
    num_epochs = 0
    loss_pt = []
    while True:
        for curve in dataset_pretrain_train:            
            curve.to(device)
            optimizer.zero_grad()

            # get loss
            loss, curve_pred, _ = ae_loss(curve, curve.curve_stats)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad)
            optimizer.step()
            
            # Store training loss values
            loss_pt.append(float(loss.detach().cpu()))
            
            if num_iters % 10 == 0:
                print(f'loss [{num_iters}/{train_cfg.max_update} iters]: {float(loss.detach().cpu())}')
                # writer.add_scalar('train/loss', loss.item(), num_iters)
            if num_iters % train_cfg.save_interval == 0:
                write_model(model, log_dir, num_iters)
            if num_iters >= train_cfg.max_update:
                write_model(model, log_dir, 'pretrained')
                return best_iter, loss_pt
            if num_iters % train_cfg.save_interval == 0:
                test(dataset_pretrain_valid, ae_loss, model, num_iters, device)
            if num_iters % 1000 == 0:
                if ae_loss.use_accum:
                    curve_pred[:, :, 1] = torch.cumsum(curve_pred[:, :, 1], dim=1)                
                curve = model.unnormalize(curve, curve_stats)
                curve_pred = model.unnormalize(curve_pred, curve_stats)
                for j, (curve_pred_single, curve_true_single) in enumerate(zip(curve_pred, curve)):                    
                    # Curves                    
                    plot({
                        'true': (to_numpy(curve_true_single)[:, 0], to_numpy(curve_true_single)[:, 1], '-.'),
                        'pred': (to_numpy(curve_pred_single)[:, 0], to_numpy(curve_pred_single)[:, 1], '-')
                    }, pn=os.path.join(log_dir, f'pretrain_iter_{num_iters}_epoch_{num_epochs}.png'))
                    break
            num_iters += 1
            del loss
        if not train_cfg.no_epoch_checkpoints:
            write_model(model, log_dir, f'epoch_{num_epochs}')
        num_epochs += 1
        print(f'iterated through whole dataset {num_epochs} times')

def test(dataset_pretrain_valid, ae_loss, model, num_iters, device):
    model.eval()
    if not os.path.isdir(os.path.join(log_dir, 'test')):
        # os.system(f"mkdir {os.path.join(log_dir, 'test')}")
        os.mkdir(os.path.join(log_dir, 'test'))
    # os.system(f"mkdir {os.path.join(log_dir, 'test', str(num_iters))}")
    os.mkdir(os.path.join(log_dir, 'test', str(num_iters)))
    y_max_pred = []
    y_max_true = []
    latent = []
    for i, curve in enumerate(tqdm(dataset_pretrain_valid)):
        # curve.curve /= torch.abs(curve.curve).max(dim=1)[0].unsqueeze(dim=1)
        with torch.no_grad():
            curve.to(device)
            curve_stats = curve.curve_stats
            curve = model.normalize(deepcopy(curve.curve), curve_stats) # (batch_size, resolution, 2)            
            loss, curve_pred, latent_i = ae_loss(curve)
            if ae_loss.use_accum:
                curve_pred[:, :, 1] = torch.cumsum(curve_pred[:, :, 1], dim=1)
        
        # Store latent vectors for each batch
        latent.append(latent_i)                
        
        curve = model.unnormalize(curve, curve_stats)
        curve_pred = model.unnormalize(curve_pred, curve_stats)
         
        y_max_pred.extend(curve_pred[:,:,1].max(dim=-1)[0].detach().cpu().tolist())
        y_max_true.extend(curve[:,:,1].max(dim=-1)[0].detach().cpu().tolist())        
        
        # Curves
        for j, (curve_pred_single, curve_true_single) in enumerate(zip(curve_pred, curve)):
            ttt = curve_true_single/(1e-12+curve_true_single.max(dim=0)[0].unsqueeze(0))
            ppp = curve_pred_single/(1e-12+curve_pred_single.max(dim=0)[0].unsqueeze(0))
            # Shape
            plot({
                'true': (
                    to_numpy(ttt)[:, 0],
                    to_numpy(ttt)[:, 1], '-.'),
                'pred': (
                    to_numpy(ppp)[:, 0],
                    to_numpy(ppp)[:, 1], '-')
            }, pn=os.path.join(log_dir, 'test', str(num_iters), f'sample_{i}-{j}_shape.png'), title='shape')
                        
            # Curve          
            plot({
                'true': (to_numpy(curve_true_single)[:, 0], to_numpy(curve_true_single)[:, 1], '-.'),
                'pred': (to_numpy(curve_pred_single)[:, 0], to_numpy(curve_pred_single)[:, 1], '-')
            }, pn=os.path.join(log_dir, 'test', str(num_iters), f'sample_{i}-{j}_curve.png'))
            break
    
    from sklearn.metrics import r2_score
    # Log magnitude (normalized curve)
    x, y = np.log(np.array(y_max_true)), np.log(np.maximum(1e-16, y_max_pred))

    z = np.polyfit(x, y, 1)
    # p = np.poly1d(z)
    r_squared = r2_score(x, y)

    plt.scatter(x, y, marker='x', label=("y=%.6fx+(%.6f) - $R^2$=%.6f" % (z[0], z[1], r_squared)))
    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    plt.grid()
    plt.plot(sorted(x), sorted(x), "y--")
    # plt.plot(x, y, 'og-', label=("y=%.6fx+(%.6f) - $R^2$=%.6f" % (z[0], z[1], r_squared)))
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'test', str(num_iters), f'log_magnitude_pred.png'))
    plt.close()
    
    # Magnitude (normalized curve)
    x, y = np.array(y_max_true), np.maximum(1e-16, y_max_pred)

    z = np.polyfit(x, y, 1)
    # p = np.poly1d(z)
    r_squared = r2_score(x, y)

    plt.scatter(x, y, marker='x', label=("y=%.6fx+(%.6f) - $R^2$=%.6f" % (z[0], z[1], r_squared)))
    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    plt.grid()
    plt.plot(sorted(x), sorted(x), "y--")
    # plt.plot(x, y, 'og-', label=("y=%.6fx+(%.6f) - $R^2$=%.6f" % (z[0], z[1], r_squared)))
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'test', str(num_iters), f'magnitude_pred.png'))
    plt.close()         

    # Latent space
    latent = torch.cat(latent, dim = 0) # (curves, d)
    tsne_plot(latent.cpu().detach().numpy(), y_max_true, y_max_pred, num_iters)     

    model.train()

def tsne_plot(latent, magnitude_true, magnitude_pred, cur_iter):
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    zis_viz = tsne.fit_transform(latent) # (n_samples, n_components)
    
    # tsne plot of latent space curve-AE: true magnitude
    plt.figure()
    plt.scatter(zis_viz[:, 0],zis_viz[:, 1], c= magnitude_true, cmap = 'viridis')
    
    norm = plt.Normalize(0, 1)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'))
    cbar.set_label('Stress magnitude')
    plt.xlabel('Comp-1')
    plt.ylabel('Comp-2')
    plt.title('Latent space curve-AE T-SNE projection')
    plt.savefig(os.path.join(log_dir, f'tsne_true_magn_{cur_iter}.png'))
    plt.clf()
    plt.close()        
   
    # tsne plot of latent space curve-AE: predicted magnitude
    plt.figure()
    plt.scatter(zis_viz[:, 0],zis_viz[:, 1], c= magnitude_pred, cmap = 'viridis')
    
    norm = plt.Normalize(0, 1)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'))
    cbar.set_label('Stress magnitude')
    plt.xlabel('Comp-1')
    plt.ylabel('Comp-2')
    plt.title('Latent space curve-AE T-SNE projection')
    plt.savefig(os.path.join(log_dir, f'tsne_pred_magn_{cur_iter}.png'))
    plt.clf()
    plt.close()     

   
def pretrain_property(dataset_pretrain_train, dataset_pretrain_valid, model,
                      optimizer, train_cfg, loss_name, device):
    model.train()
    model = model.to(device)
    regr_loss = \
        RegressLoss(
            model.encoder, model.decoder.dim_hidden, loss_name, device
            )

    best_iter = -1
    num_iters = 0
    num_epochs = 0
    while True:
        for graph, prop in dataset_pretrain_train:            
            graph.to(device), prop.to(device)
            optimizer.zero_grad()

            # get loss           
            loss, _ = regr_loss(graph, prop)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad)
            optimizer.step()
            if num_iters % 10 == 0:
                print(f'loss [{num_iters}/{train_cfg.max_update} iters]: {float(loss.detach().cpu())}')
                # writer.add_scalar('train/loss', loss.item(), num_iters)
            if num_iters % train_cfg.save_interval == 0:
                write_model(model, log_dir, num_iters)
            if num_iters >= train_cfg.max_update:
                write_model(model, log_dir, 'pretrained')
                return best_iter
            if num_iters % 5000 == 0:
                test_prop(dataset_pretrain_valid, regr_loss, model, num_iters, device)
                
            num_iters += 1
            del loss
        if not train_cfg.no_epoch_checkpoints:
            write_model(model, log_dir, f'epoch_{num_epochs}')
        num_epochs += 1
        print(f'iterated through whole dataset {num_epochs} times')

def test_prop(dataset_pretrain_valid, regr_loss, model, num_iters, device):
    model.eval()
    if not os.path.isdir(os.path.join(log_dir, 'test')):
        # os.system(f"mkdir {os.path.join(log_dir, 'test')}")
        os.mkdir(os.path.join(log_dir, 'test'))
    # os.system(f"mkdir {os.path.join(log_dir, 'test', str(num_iters))}")
    os.mkdir(os.path.join(log_dir, 'test', str(num_iters)))
    prop_li = []
    prop_pred_li = []
    for i, (graph, prop) in enumerate(tqdm(dataset_pretrain_valid)):        
        with torch.no_grad():
            graph.to(device), prop.to(device)           
            loss, prop_pred = regr_loss(graph, prop)           
                
        prop_li.append(prop.prop.cpu().detach().numpy()) # (batch_size, 1)
        prop_pred_li.append((regr_loss.unnormalize(prop_pred, prop.prop_stats)).cpu().detach().numpy())
        
    from sklearn.metrics import r2_score
    # Log of the property
    x, y = np.log(np.array(prop_li)), np.array(prop_pred_li)
    x = x.reshape(-1)
    y = y.reshape(-1)
    
    z = np.polyfit(x, y, 1)
    # p = np.poly1d(z)
    r_squared = r2_score(x, y)

    plt.scatter(x, y, marker='x', label=("y=%.6fx+(%.6f) - $R^2$=%.6f" % (z[0], z[1], r_squared)))
    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    plt.grid()
    plt.plot(sorted(x), sorted(x), "y--")
    # plt.plot(x, y, 'og-', label=("y=%.6fx+(%.6f) - $R^2$=%.6f" % (z[0], z[1], r_squared)))
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'test', str(num_iters), f'log_property_pred.png'))
    plt.close()
    
    # Property
    x, y = np.array(prop_li), np.exp(prop_pred_li)
    x = x.reshape(-1)
    y = y.reshape(-1)
    
    z = np.polyfit(x, y, 1)
    # p = np.poly1d(z)
    r_squared = r2_score(x, y)

    plt.scatter(x, y, marker='x', label=("y=%.6fx+(%.6f) - $R^2$=%.6f" % (z[0], z[1], r_squared)))
    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    plt.grid()
    plt.plot(sorted(x), sorted(x), "y--")
    # plt.plot(x, y, 'og-', label=("y=%.6fx+(%.6f) - $R^2$=%.6f" % (z[0], z[1], r_squared)))
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'test', str(num_iters), f'property_pred.png'))
    plt.close()   
    
    model.train()

class RegressLoss(torch.nn.Module):
    def __init__(self, encoder, dim_hidden, loss_name, device):
        super(RegressLoss, self).__init__()
        self.encoder = encoder
        self.decoder = PropertyDecoder(dim_hidden).to(device)
        self.loss_name = loss_name
        
    def forward(self, graph, prop_true):              
        emb_nodes, emb_edges = self.encoder(graph.feats_node, graph.feats_edge, graph.edge_index)                
        prop_pred = self.decoder(emb_nodes, emb_edges, graph)      
        prop_true = self.normalize(prop_true.prop, prop_true.prop_stats)        
        
        if self.loss_name == 'mse':
            loss = \
                F.mse_loss(
                    prop_pred,
                    prop_true
                )
        elif self.loss_name == 'huber':
            loss = \
                F.huber_loss(
                    prop_pred,
                    prop_true
                )
        else:
            assert False
        
        return loss, prop_pred
    
    def normalize(self, prop, stats):       
        prop = (prop - stats['mean']) / stats['std']
        return prop
    
    def unnormalize(self, prop, stats):               
        prop = prop*stats['std'] + stats['mean']
        return prop

def read_model(model, dn, run_name):
    model.load_state_dict(torch.load(os.path.join(dn, f'model_{run_name}.pt')))

def write_model(model, dn, run_name):
    torch.save(model.state_dict(), os.path.join(dn, f'model_{run_name}.pt'))


class PropertyDecoder(torch.nn.Module):
    def __init__(self, dim_hidden):
        super(PropertyDecoder, self).__init__()
        self.pool_nodes = GatedAttentionPooling(64)
        self.pool_edges = GatedAttentionPooling(64)
        self.merger = CatMerger(64, dim_hidden)
        self.out_mlp = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.ELU(),
            nn.Linear(dim_hidden, dim_hidden), nn.ELU(),
            nn.Linear(dim_hidden, 1)
        )                        
        self.bn = nn.BatchNorm1d(dim_hidden)     
        
    def forward(self, emb_nodes, emb_edges, graph):
        emb_all = self.get_pooled_emb(emb_nodes, emb_edges, graph)               
        prop = self.out_mlp(emb_all)       
        return prop

    def get_pooled_emb(self, emb_nodes, emb_edges, graph):
        emb_nodes_pooled = \
            self.pool_nodes(
                src=emb_nodes, index=graph.graph_node_index,
                dim=0, dim_size=len(graph.g_li)
            )
        emb_edges_pooled = \
            self.pool_edges(
                src=emb_edges, index=graph.graph_edge_index,
                dim=0, dim_size=len(graph.g_li)
            )
        emb_all = self.merger(emb_nodes_pooled, emb_edges_pooled)
        if self.bn is not None:
            emb_all = self.bn(emb_all)        
        return emb_all

class CurveEncoder(torch.nn.Module):
    def __init__(self, dim_hidden, decoder_cfg, max_value=50):
        super(CurveEncoder, self).__init__()
        self.cpos_enc = ContinuousPositionalEncoding(dim_hidden//2, max_value=max_value)
        self.cnn = \
            nn.Sequential(
                nn.ConvTranspose1d(dim_hidden, dim_hidden, kernel_size=3, stride=1, padding=0),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
                nn.ConvTranspose1d(dim_hidden, dim_hidden, kernel_size=3, stride=1, padding=0),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
                nn.ConvTranspose1d(dim_hidden, dim_hidden, kernel_size=3, stride=1, padding=0),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
                nn.ConvTranspose1d(dim_hidden, dim_hidden, kernel_size=3, stride=1, padding=0),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
                nn.ConvTranspose1d(dim_hidden, dim_hidden, kernel_size=3, stride=1, padding=0),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
                # nn.ConvTranspose2d(dim_hidden, dim_hidden, kernel_size=(3, 1), stride=stride, padding=1)
            ) 
        # self.lstm = RNNs(dim_hidden, **decoder_cfg['args']['rnn_args'])
        # self.gru = torch.nn.GRU(
        #     input_size=2,
        #     hidden_size=dim_hidden,
        #     num_layers=3,
        #     batch_first=True
        # )
        self.embeddings = torch.randn(2, dim_hidden)
        self.existing_curves = []
        # self.out_mlp = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ELU())
        self.out_mlp = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.ELU(),
            nn.Linear(dim_hidden, dim_hidden), nn.ELU(),
            nn.Linear(dim_hidden, dim_hidden)
        )
        self.bn = nn.BatchNorm1d(dim_hidden)

    def forward(self, curve):
        x = torch.tensor(np.concatenate((
            self.cpos_enc(curve[:,:,0].unsqueeze(-1).cpu().detach().numpy()),
            self.cpos_enc(curve[:,:,1].unsqueeze(-1).cpu().detach().numpy())), axis=-1), device=curve.device)
        x = self.cnn(x.transpose(1,2)).transpose(1,2)
        # x = self.lstm(x)
        # x, _ = self.gru(curve)
        x = torch.mean(x, dim=1)
        x = self.out_mlp(x)
        x = self.bn(x)
        return x

class AELoss(torch.nn.Module):
    def __init__(self, dim_hidden, decoder_cfg, decoder, loss_name, use_accum, device):
        super(AELoss, self).__init__()
        self.encoder = CurveEncoder(dim_hidden, decoder_cfg).to(device)
        self.decoder = decoder
        self.loss_name = loss_name
        self.use_accum = use_accum

    def forward(self, curve_true):
        latent = self.encoder(curve_true)
        latent = F.normalize(latent, p=2)

        stress_max = strain.max(dim=-1)[0]
        curve_pred, other = self.decoder(latent, graph, stress_max)

        strain, stress_true_shape, stress_true_magnitude = \
            model_surrogate.split_stress_strain(curve.curve)
        strain, _ = \
            model_surrogate.norm_stress_strain(strain, stress_true_magnitude, curve_stats=curve.curve_stats)
        curve_pred, _ = model_surrogate.inference(graph, strain=strain, curve_stats=curve.curve_stats)

        # latent = F.layer_norm(latent)
        curve_pred, other = self.decoder(latent, strain_max=strain_max)

        if self.loss_name == 'disentangle':
            assert other is not None and 'shape' in other and 'magnitude' in other
            
            # Ground-truth
            # curve_shape_true = curve_true/curve_true[:,-1,:].unsqueeze(1)
            # curve_mag_true = curve_true[:,-1]
            curve_shape_true = curve_true/curve_true.max(dim = -2)[0].unsqueeze(1)       
            curve_mag_true = curve_true.max(dim = -2)[0]
            
            # Prediction
            curve_shape_pred = other['shape']
            curve_mag_pred = other['magnitude']
            if self.use_accum:
                y_accum = \
                    torch.cat((
                        torch.zeros(curve_shape_true.shape[0], 1, device=curve_shape_true.device),
                        torch.cumsum(curve_shape_pred[:, :-1, 1], dim=1).detach()
                        # curve_true[:,:-1,1]#torch.cumsum(curve_true[:,:,1], dim=1)[:,:-1]
                    ), dim=1).detach()
                curve_shape_true[:, :, 1] -= y_accum
            loss = \
                F.huber_loss(
                    curve_shape_pred.view(-1, curve_shape_pred.shape[-1]),
                    curve_shape_true.view(-1, curve_shape_true.shape[-1])
                ) + 5 * F.huber_loss(curve_mag_pred, torch.log(curve_mag_true))
            # loss = \
            #     F.mse_loss(
            #         curve_shape_pred.view(-1, curve_shape_pred.shape[-1]),
            #         curve_shape_true.view(-1, curve_shape_true.shape[-1])
            #     )             
        else:
            if self.use_accum:
                y_accum = \
                    torch.cat((
                        torch.zeros(curve_true.shape[0], 1, device=curve_true.device),
                        torch.cumsum(curve_pred[:, :-1, 1], dim=1).detach()
                        # curve_true[:,:-1,1]#torch.cumsum(curve_true[:,:,1], dim=1)[:,:-1]
                    ), dim=1).detach()
                curve_true[:, :, 1] -= y_accum
            if self.loss_name == 'mse':
                loss = \
                    F.mse_loss(
                        curve_pred.view(-1, curve_pred.shape[-1]),
                        curve_true.view(-1, curve_true.shape[-1])
                    )
            elif self.loss_name == 'huber':
                loss = \
                    F.huber_loss(
                        curve_pred.view(-1, curve_pred.shape[-1]),
                        curve_true.view(-1, curve_true.shape[-1])
                    )
            else:
                assert False
        return loss, curve_pred, latent

class NTXentLoss(torch.nn.Module):
    def __init__(self, device, temperature=0.1, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = None
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    # def forward(self, zis, zjs):
    #     assert zis.shape[0] == zjs.shape[0]
    #     if zis.shape[0] != self.batch_size:
    #         self.batch_size = zis.shape[0]
    #         self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
    #     representations = torch.cat([zjs, zis], dim=0)
    #
    #     similarity_matrix = self.similarity_function(representations, representations)
    #
    #     # filter out the scores from the positive samples
    #     l_pos = torch.diag(similarity_matrix, self.batch_size)
    #     r_pos = torch.diag(similarity_matrix, -self.batch_size)
    #     positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
    #
    #     negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
    #
    #     logits = torch.cat((positives, negatives), dim=1)
    #     logits /= self.temperature
    #
    #     labels = torch.zeros(2 * self.batch_size).to(self.device).long()
    #     loss = self.criterion(logits, labels)
    #
    #     return loss / (2 * self.batch_size)


    # def forward(self, features, labels=None, mask=None):
    def forward(self, zis, zjs, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = torch.stack([zis, zjs], dim=1)
        features = F.normalize(features, dim=-1)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1e-12)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
