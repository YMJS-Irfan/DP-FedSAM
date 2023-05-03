import copy
import logging
import pickle
import random
import pdb
import numpy as np
import torch

from fedml_api.model.cv.resnet import  customized_resnet18
from fedml_api.dpfedsam.client import Client
import os

class DPFedSAMAPI(object):
    def __init__(self, dataset, device, args, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.init_stat_info()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, self.logger)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    def train(self, exper_index):
        w_global = self.model_trainer.get_model_params()
        nabala_w_global = copy.deepcopy(subtract(w_global, w_global ))

        self.logger.info("################Exper times: {}".format(exper_index))
        for round_idx in range(self.args.comm_round):
            w_locals = []
            w_locals, nabala_w = [], []
            last_w_global = copy.deepcopy(w_global)
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            """
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                self.args.client_num_per_round)
            client_indexes = np.sort(client_indexes)

            # self.logger.info("client_indexes = " + str(client_indexes))
            loss_locals, acc_locals, total_locals = [], [], []

            norm_list = []
            for cur_clnt in client_indexes:
                # update dataset
                client = self.client_list[cur_clnt]
                # local model training
                w_per,training_flops,num_comm_params, metrics = client.train(copy.deepcopy(w_global), round_idx)
                # calculate the local update of each participated client
                nabala = copy.deepcopy(subtract(w_per, w_global))
                # initialize the norm 
                norm = 0.0
                ###### add client-level DP noise ##############
                for name in nabala.keys():
                    norm += pow(nabala[name].norm(2), 2)
                    noise = torch.FloatTensor(nabala[name].shape).normal_(0, self.args.sigma * self.args.C /np.sqrt(self.args.client_num_per_round))
                    noise = noise.cpu().numpy()
                    noise = torch.from_numpy(noise).type(torch.FloatTensor).to(nabala[name].device)
                    nabala[name] *= min(1, self.args.C/torch.norm(nabala[name], 2))
                    nabala[name] = nabala[name].add_(noise)

                total_norm = np.sqrt(norm).numpy().reshape(1)
                # print(total_norm[0])
                norm_list.append(total_norm[0])

                w_per = copy.deepcopy(add(nabala, w_global))
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w_per)))
                nabala_w.append((client.get_sample_number(), nabala))
                self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params
                loss_locals.append(metrics['train_loss'])
                acc_locals.append(metrics['train_correct'])
                total_locals.append(metrics['train_total'])


            self.stat_info["local_norm"].append(norm_list)
            global_norm = sum(norm_list)/len(norm_list)
            self.stat_info["global_norm"].append(global_norm)
            # print('global_norm = {}'.format(self.stat_info["global_norm"]))
            # print('local_norm = {}'.format(self.stat_info["local_norm"]))
            

            self._train_on_sample_clients(loss_locals, acc_locals, total_locals, round_idx, len(client_indexes))
            # update global meta weights
            nabala_w_global = self._aggregate(nabala_w)
            w_global = copy.deepcopy(add(last_w_global, nabala_w_global))

            
            self._test_on_all_clients(w_global, round_idx)

            if round_idx % 50 == 0 or round_idx == self.args.comm_round -1 :
                print('global_train_loss={}'.format(self.stat_info["global_train_loss"]))
                print('global_train_acc={}'.format(self.stat_info["global_train_acc"]))
                print('global_test_loss={}'.format(self.stat_info["global_test_loss"]))
                print('global_test_acc={}'.format(self.stat_info["global_test_acc"]))
                self.logger.info("################Communication round : {}".format(round_idx))
                if round_idx % 200 == 0 or round_idx == self.args.comm_round-1:
                    self.logger.info("################The final results, Experiment times: {}".format(exper_index))
                    # self.logger.info('local_norm = {}'.format(self.stat_info["local_norm"]))
                    np.array(self.stat_info["local_norm"]).dump("./LOG/cifar10/dumps/local_norm_dpfedsam_{self.args.p}_.dat")
                    if self.args.dataset ==  "cifar10":
                        model = customized_resnet18(10)
                        model.load_state_dict(copy.deepcopy(w_global)) 
                        if self.args.spar_rand == True:
                            torch.save(model,f"{os.getcwd()}/save_model/dp-fedsam_threshold{self.args.C}_rho{self.args.rho}_spar_rand_p{self.args.p}.pth.tar")
                        else:
                            torch.save(model,f"{os.getcwd()}/save_model/dp-fedsam_threshold{self.args.C}_rho{self.args.rho}_spar_topk_p{self.args.p}.pth.tar")

                self.logger.info('local_norm = {}'.format(self.stat_info["local_norm"]))
                self.logger.info('global_norm = {}'.format(self.stat_info["global_norm"]))
                self.logger.info('global_train_loss={}'.format(self.stat_info["global_train_loss"]))
                self.logger.info('global_train_acc={}'.format(self.stat_info["global_train_acc"]))
                self.logger.info('global_test_loss={}'.format(self.stat_info["global_test_loss"]))
                self.logger.info('global_test_acc={}'.format(self.stat_info["global_test_acc"]))

                
        


    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num
        w_global ={}
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    w_global[k] = local_model_params[k] * w
                else:
                    w_global[k] += local_model_params[k] * w
        return w_global

    def _train_on_sample_clients(self, loss_locals, acc_locals, total_locals, round_idx, client_sample_number):
        self.logger.info("################global_train_on_all_clients : {}".format(round_idx))

        g_train_acc = sum([np.array(acc_locals[i]) / np.array(total_locals[i]) for i in
                        range(client_sample_number)]) / client_sample_number
        g_train_loss = sum([np.array(loss_locals[i]) / np.array(total_locals[i]) for i in
                         range(client_sample_number)]) / client_sample_number

        print('The averaged global_train_acc:{}, global_train_loss:{}'.format(g_train_acc, g_train_loss))
        stats = {'The averaged global_train_acc': g_train_acc, 'global_train_loss': g_train_loss}
        self.stat_info["global_train_acc"].append(g_train_acc)
        self.stat_info["global_train_loss"].append(g_train_loss)
        self.logger.info(stats)


    def _test_on_all_clients(self, w_global, round_idx):

        self.logger.info("################global_test_on_all_clients : {}".format(round_idx))
        g_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        for client_idx in range(self.args.client_num_in_total):
            client = self.client_list[client_idx]
            g_test_local_metrics = client.local_test(w_global, True)
            g_test_metrics['num_samples'].append(copy.deepcopy(g_test_local_metrics['test_total']))
            g_test_metrics['num_correct'].append(copy.deepcopy(g_test_local_metrics['test_correct']))
            g_test_metrics['losses'].append(copy.deepcopy(g_test_local_metrics['test_loss']))


            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on test dataset
        g_test_acc = sum([np.array(g_test_metrics['num_correct'][i]) / np.array(g_test_metrics['num_samples'][i]) for i in
                        range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        g_test_loss = sum([np.array(g_test_metrics['losses'][i]) / np.array(g_test_metrics['num_samples'][i]) for i in
                         range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        stats = {'global_test_acc': g_test_acc, 'global_test_loss': g_test_loss}
        self.stat_info["global_test_acc"].append(g_test_acc)
        self.stat_info["global_test_loss"].append(g_test_loss)
        self.logger.info(stats)


    def record_avg_inference_flops(self, w_global, mask_pers=None):
        inference_flops = []
        for client_idx in range(self.args.client_num_in_total):

            if mask_pers == None:
                inference_flops += [self.model_trainer.count_inference_flops(w_global)]
            else:
                w_per = {}
                for name in mask_pers[client_idx]:
                    w_per[name] = w_global[name] * mask_pers[client_idx][name]
                inference_flops += [self.model_trainer.count_inference_flops(w_per)]
        avg_inference_flops = sum(inference_flops) / len(inference_flops)
        self.stat_info["avg_inference_flops"] = avg_inference_flops

    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["global_train_acc"] = []
        self.stat_info["global_train_loss"] = []
        self.stat_info["global_test_acc"] = []
        self.stat_info["global_test_loss"] = []
        self.stat_info["person_test_acc"] = []
        self.stat_info["final_masks"] = []
        self.stat_info["global_norm"] = []
        self.stat_info["local_norm"] = []



def subtract(params_a, params_b):
        w = copy.deepcopy(params_a)
        for k in w.keys():
                w[k] -= params_b[k]
        return w

def add(params_a, params_b):
        w = copy.deepcopy(params_a)
        for k in w.keys():
                w[k] += params_b[k]
        return w