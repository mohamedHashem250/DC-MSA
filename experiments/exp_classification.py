from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy, cal_metrics
import torch
import torch.nn as nn
from torch import optim
import pandas as pd
import torch
import os
import time
import warnings
import numpy as np
import pdb
import torch.utils.benchmark as benchmark
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()  #self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze())
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy
    
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
    
    def train(self, setting):
        #dataset_name = os.path.splitext(self.args.root_path)[0]
        dataset_name = os.path.basename(os.path.normpath(self.args.root_path))
        print('dataset_name:', dataset_name)
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        #path = os.path.join(self.args.checkpoints, setting)
        path = os.path.join( dataset_name, self.args.model, self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        print('path: ', path)
        time_now = time.time()
        print("Train Starting..............................................................")
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        total_train_losses = []
        total_val_losses = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                #print(i, " I am training now!!!!")
                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)
            total_train_losses.append(train_loss)
            total_val_losses.append(vali_loss)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.save_losses( parent_path = path, train_losses = total_train_losses, val_losses= total_val_losses)
        return self.model


    #count learnable parameters: 
    def count_parameters(self) -> int: #, model: nn.Module)
        """
        Frozen parameters (requires_grad=False) are included.
        """
        return sum(p.numel() for p in self.model.parameters() )#if p.requires_grad
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
        dataset_name = os.path.basename(os.path.normpath(self.args.root_path))
        #dataset_name = os.path.splitext(self.args.data_path)[0]
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        #folder_path = './test_results/' + setting + '/'
        folder_path = os.path.join(dataset_name,self.args.model, 'test_results', setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)
        print("batch_x.shape: ", batch_x.shape)
        print("padding_mask.shape: ",padding_mask.shape)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        #accuracy = cal_accuracy(predictions, trues)
        accuracy, precision, recall, f1 = cal_metrics(predictions, trues)

        #can we calculate other metrics like Recall , Precision, F1-Score
        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # save for ROC/AUC plotting later
        probs_np = probs.cpu().numpy()      # (total_samples, num_classes) — probability scores
        trues_np = trues                     # already numpy from above

        np.save(os.path.join(folder_path, 'probs.npy'), probs_np)   # softmax probabilities
        np.save(os.path.join(folder_path, 'trues.npy'), trues_np)   # true integer labels
        np.save(os.path.join(folder_path, 'preds.npy'), predictions) # predicted class (optional)


        print('accuracy:{:.4f}'.format(accuracy))
        print('precision:{:.4f}'.format(precision))
        print('recall:{:.4f}'.format(recall))
        print('f1:{:.4f}'.format(f1))
        #memory size and model parameters::
        lr_params = self.count_parameters()
        mem_size = self.model_memory_mb()
        print("model learnable parameters: ",lr_params)
        print("model size: ", mem_size )
        #inference time: 
        #calculate the size of dummy input
        dummy_batch_size = 1
        dummy_batch_x = torch.randn(dummy_batch_size , batch_x.shape[-2], batch_x.shape[-1]).to(self.device)
        dummy_padding_mask = torch.randn(dummy_batch_size , padding_mask.shape[-1]).to(self.device)
        print("dummy_batch_x.shape: ", dummy_batch_x.shape)
        print("dummy_padding_mask.shape: ",dummy_padding_mask.shape)
        dummy_x_mark = None
        dummy_y_mark = None

        inps =  [dummy_batch_x, dummy_padding_mask, None, None]
      
        inf_time = self.inference_time( dummy_input= inps)
        print("inferenece time: ", inf_time)
        macs, params = profile(self.model, inputs=(dummy_batch_x, dummy_padding_mask, None, None),verbose=True)

        print(f"MACs: {macs / 1e6:.2f} G")
        print(f"Params: {params / 1e6:.2f} M")
        print('with Fvcore::::::::::::')
        self.model.eval()

        inputs = (
                    dummy_batch_x,
                    dummy_padding_mask,
                    None, 
                    None           
        )
        try:
            flops = FlopCountAnalysis(self.model, inputs)
            print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
        except Exception as e:
            print(f"FVCore FLOPs skipped: {e}")


        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write(f'accuracy:{accuracy:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}')
        f.write('\n')
        f.write('learnable params:{}, mmerory size: [learanble, buffer, total]:{}'.format(lr_params, mem_size))
        f.write('\n')
        f.write('inferenece time: {} ms'.format(inf_time))
        f.write('\n')
        f.write('MACs: {} params {} from flops'.format(macs/1e6, params/1e6))
        f.write('\n')
        f.close()
        return