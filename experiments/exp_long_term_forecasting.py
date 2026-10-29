from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.utils.benchmark as benchmark
warnings.filterwarnings('ignore')
from thop import profile

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model



    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def save_losses(self, parent_path, train_losses, val_losses):
        #save the train,and validation losses in csv and npy files for later usage.
        #check len 
        if len(train_losses) != len(val_losses):
            raise ValueError(" len must be equal")
        
        save_path = os.path.join(parent_path, 'loss_recordings')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #save the losses in file to compare: 
        #the best way csv or npy let's make the both:
        loss_record = pd.DataFrame( {
            'epochs':np.arange(1, len(train_losses)+1),
            'train_loss':np.array(train_losses,dtype = np.float32),
            'val_loss':np.array(val_losses, dtype = np.float32),
        })
        loss_record.to_csv(os.path.join(save_path,'train_vali_loss.csv'),index= False )
        #save a npy file :
        np.save(os.path.join(save_path, 'train_vali_loss.npy'), loss_record.values)
        print("loss records saved!!")
        return
    

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    #count learnable parameters: 
    def count_parameters(self) -> int: #, model: nn.Module)
        """
        Frozen parameters (requires_grad=False) are included.
        """
        self.model.eval()
        return sum(p.numel() for p in self.model.parameters() )#if p.requires_grad
    
    def train(self, setting):
        lr_params = self.count_parameters()
        print("model learnable parameters: ",lr_params)
        dataset_name = os.path.splitext(self.args.data_path)[0]
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.model, dataset_name, self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        total_train_losses = []
        total_val_losses = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)

            total_train_losses.append(train_loss)
            total_val_losses.append(vali_loss)
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                     epoch + 1, train_steps, train_loss, vali_loss))
            #self.test(setting)  ######################################################
            # #########################################################################
            # #########################################################################      
            
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.save_losses( parent_path = path, train_losses = total_train_losses, val_losses= total_val_losses)
        return self.model


    #memory size ==> learnable parameters, and buffers 

    def model_memory_mb(self) -> float:
        """
        Estimate storage size in megabytes.

        Includes parameters AND registered buffers.
        Buffers are non-trainable tensors in state_dict, e.g.:
        PositionalEncoding.pe  — the sinusoidal table (max_len * d_model floats)

        p.element_size() = bytes per scalar: 4 for float32, 2 for float16.
        """
        params_size , buffers_size = 0,0
        for p in self.model.parameters():
            params_size += p.numel() * p.element_size()
        for b in self.model.buffers():
            buffers_size += b.numel() * b.element_size()
        params_size /= (1024 ** 2)
        buffers_size /= (1024 ** 2)
        return params_size , buffers_size , params_size + buffers_size

    def inference_time(self, dummy_input, num_runs=100):

        self.model.eval()

        # Warm-up
        with torch.no_grad():
            for _ in range(20):

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        self.model(*dummy_input)
                else: 
                 self.model(*dummy_input)

        if dummy_input[0].is_cuda:
            torch.cuda.synchronize()

        t = benchmark.Timer(
            stmt='''
if use_amp:
    with torch.cuda.amp.autocast():
        model(*x)
else:
    model(*x)
if x[0].is_cuda:
    torch.cuda.synchronize()
''',
            globals={
                'model': self.model,
                'x': dummy_input,
                'torch': torch,
                'use_amp': self.args.use_amp
            }
        )

        with torch.no_grad():
            result = t.timeit(num_runs)

        latency_ms = result.mean * 1000 #to produce the result in ms

        return latency_ms

    def test(self, setting, test=0):
        dataset_name = os.path.splitext(self.args.data_path)[0]
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.model, dataset_name, self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = os.path.join(self.args.model, dataset_name, 'test_results', setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device, non_blocking = True)
                batch_y = batch_y.float().to(self.device, non_blocking = True)# ground truth 

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device, non_blocking = True)
                    batch_y_mark = batch_y_mark.float().to(self.device, non_blocking = True)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device, non_blocking = True)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device, non_blocking = True)
                outputs = outputs.detach().cpu().numpy()# from model ( prediction )
                batch_y = batch_y.detach().cpu().numpy() # groud truth
                if test_data.scale and self.args.inverse:# the inverse should be works when the scale is exisit!!!
                    # this part is we want to calculate the metrics on unnormalized version of the output (univeriate time sereis)in our case SML-index
                    shape = outputs.shape
                    if self.args.features == 'MS' and outputs.shape[-1] ==1: 
                        target_index = -1
                        #squeeze function will take affect here because the batch size in test function is equal 1. 
                        outputs = (outputs.squeeze(0) *test_data.scaler.scale_[target_index] + test_data.scaler.mean_[target_index]).reshape(shape)
                        batch_y = (batch_y.squeeze(0)*test_data.scaler.scale_[target_index] + test_data.scaler.mean_[target_index]).reshape(shape)                      
                    else: 
                        
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                #we can do that later!!!
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join( folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # # result save
        # folder_path = os.path.join(self.args.model, 'test_results', setting) + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        mae, mse, rmse, mape, mspe, r_s , nrmse, smape  = metric(preds, trues)
        print('mse:{}, mae:{}, r_2: {}'.format(mse, mae, r_s))
        lr_params = self.count_parameters()
        mem_size = self.model_memory_mb()
        print("model learnable parameters: ",lr_params)
        print("model size: ", mem_size )
        #inference time: 
        #calculate the size of dummy input
     
        dummy_batch_size = 1
        dummy_batch_x = torch.randn(dummy_batch_size , batch_x.shape[-2], batch_x.shape[-1]).to(self.device, non_blocking = True)
        dummy_dec_inp = torch.randn(dummy_batch_size , dec_inp.shape[-2], dec_inp.shape[-1]).to(self.device, non_blocking = True)
        dummy_x_mark = None
        dummy_y_mark = None
        if  'PEMS' not in self.args.data or 'Solar' not in self.args.data:
            dummy_x_mark =  torch.randn(dummy_batch_size , batch_x_mark.shape[-2], batch_x_mark.shape[-1]).to(self.device, non_blocking = True)
            dummy_y_mark =  torch.randn(dummy_batch_size , batch_y_mark.shape[-2], batch_y_mark.shape[-1]).to(self.device, non_blocking = True)

        inps =  [dummy_batch_x, dummy_x_mark, dummy_dec_inp, dummy_y_mark]
      
        inf_time = self.inference_time( dummy_input= inps)
        print("inferenece time: ", inf_time)
        #calculate Flops
        macs, params = profile(self.model, inputs=(dummy_batch_x, dummy_x_mark, dummy_dec_inp, dummy_y_mark,),verbose=True)

        print(f"MACs: {macs / 1e6:.2f} G")
        print(f"Params: {params / 1e6:.2f} M")
        print('with Fvcore::::::::::::')
        self.model.eval()

        inputs = (
                        dummy_batch_x,
                        dummy_x_mark,
                        dummy_dec_inp,
                        dummy_y_mark            
        )
        #use FVCore 
        try: 
            flops = FlopCountAnalysis(self.model, inputs)

            print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
        except Exception as e:
            print(f'FVCore Flops skipped: {e}')# sometimes this library fails for example with Informer
        os.makedirs(os.path.join(self.args.model, dataset_name), exist_ok=True)
        f = open(os.path.join(self.args.model, dataset_name, "result_long_term_forecast.txt"), 'a')
        f.write(setting + "  \n")
        f.write('mse:{0:0.5f}, mae:{1:0.5f}, r_2: {2:0.5f} rmse:{3:0.5f},  mape{4:0.5f}, mspe{5:0.5f} nrmse {6:0.5f}, smape{7:0.5f}'.format(mse, mae, r_s,  rmse, mape, mspe, nrmse, smape ))
        f.write('\n')
        f.write('learnable params:{}, mmerory size: [learanble, buffer, total]:{}'.format(lr_params, mem_size))
        f.write('\n')
        f.write('inferenece time: {} ms'.format(inf_time))
        f.write('\n')
        f.write('MACs: {} params {} from flops'.format(macs/1e6, params/1e6))
        # f.write('\n')
        # f.write('flops: {} from fvcpre'.format(flops /1e6))
        # f.write('\n')
        f.write('\n')
        f.close()
        results_path = os.path.join(self.args.model, dataset_name, 'results', setting) + '/'
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        np.save(results_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r_s]))
        np.save(results_path + 'metrics_performance.npy', np.array([inf_time, macs, params]))
        np.save(results_path + 'pred.npy', preds)
        np.save(results_path+ 'true.npy', trues)

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.model,self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return