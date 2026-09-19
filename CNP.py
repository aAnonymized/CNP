import torch
import timm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
import tqdm
import datetime
import os
import random
from sampler import get_sampler
from dataset.get_datasets import get_dataloaders, get_datasets
from models.get_model import get_model
from optimizers.get_optimizer import get_optimizer, get_scheduler
from losses.get_loss import get_loss_functions, calculate_loss
from losses.MixUp import mixup_data
from utils.log_accuracy import *
from utils.setup_configs import *
from utils.utils import *
from models.KNNClassifier import KNNClassifier

# loss
from losses.balanced_label_informax_loss import DynamicDecodableInfoMaxLoss
from losses.dual_pattern_relevant_nuisance_resnet import DualPatternResNet
from utils.get_prior import get_prior
from utils.cal_margin import compare_gt_competitor_margin
# from utils.select_nrg import select_nrg_tau_with_banks


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def main():
    configs = setup_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = configs.cuda.gpu_id
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(configs.general.seed)
    print(configs)
    best_acc = 0.0
    save_name = '%s_%s_%s_%s_%s_%s_%s_%s_%s' %(
                                                configs.datasets.imbalance_ratio, 
                                                configs.general.method, 
                                                configs.general.loss_function,
                                                configs.general.img_size,
                                                configs.model.model_name,
                                                configs.model.pretrained,
                                                configs.datasets.batch_size,
                                                configs.general.seed,
                                                configs.datasets.transforms.train
                                                )
    configs.general.save_name = save_name
    outputs_dir = 'outputs/%s/%s/' %(configs.general.dataset_name, configs.general.save_name)
    datasets = get_datasets(configs)
    cls_num_list = get_cls_num_list(datasets['train'])
    print(cls_num_list)
    configs.datasets.cls_num_list = cls_num_list
    loss_functions =  get_loss_functions(configs, cls_num_list)
    dataloaders = get_dataloaders(datasets, configs)
    configs.general.num_classes = len(cls_num_list)
    model = timm.create_model(model_name=configs.model.model_name, pretrained=configs.model.pretrained, \
                              num_classes=configs.general.num_classes)
    if configs.model.local_pretrained:
        if configs.general.stage == 'IM':
            if 'outputs' in configs.model.IM_pretrained_path:
                state_dict = torch.load(configs.model.IM_pretrained_path).state_dict()
            else:
                state_dict = torch.load(configs.model.IM_pretrained_path)
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
            state_dict.pop('classifier.weight', None)
            state_dict.pop('classifier.bias', None)
        elif configs.general.stage == 'Purification':
            if 'outputs' in configs.model.Purify_pretrained_path:
                state_dict = torch.load(configs.model.Purify_pretrained_path).state_dict()  
            else:
                state_dict = torch.load(configs.model.Purify_pretrained_path)
        else:
            print(f'Error !! Pls ensure configs.general.stage == [IM | Purification].')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f'load student model from local, missing: {missing}; unexpected: {unexpected}')
    if configs.cuda.use_gpu:
        model = model.cuda()
        # tau_star = select_nrg_tau_with_banks(
        #     model=model,
        #     val_loader=dataloaders['val'],
        #     prototype_k_factors=[2, 4],
        #     thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        #     verbose=True,
        # )["best_tau"]
        if configs.general.stage == 'Purification':
            rn_method = DualPatternResNet(
                num_classes=configs.general.num_classes,
                prototype_k_factors=[2, 4],
                decision_threshold=0.1, # 0.7, 0.6, 0.4
                class_counts=cls_num_list,
                clustering_metric="cosine",
            )
        else:
            pass

    device = next(model.parameters()).device
    train_epochs = 0
    if configs.general.stage == 'IM':
        prior_log, _ = get_prior(cls_num_list, device)
        train_epochs = configs.general.IM_epochs
        if os.path.exists(outputs_dir):
            [os.remove(os.path.join(outputs_dir, file_name)) for file_name in os.listdir(outputs_dir)]
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
    elif configs.general.stage == 'Purification':
        train_epochs = configs.general.Purify_epochs
        classifier = model.get_classifier()
        for param in classifier.parameters():
            param.requires_grad = False
        discovery_result = rn_method.discover(
            model=model,
            reference_loader=dataloaders['val'],
            device=device,
            verbose=True,
        )
    else:
        print(f'Error !! Pls ensure configs.general.stage == [IM | Purification].')
    optimizer = get_optimizer(configs, model)
    
<<<<<<< HEAD
    if configs.general.test:
        result = rn_method.evaluate_bank_nuisance_removal(
            student_model=model,
            inputs=dataloaders["test"],
            save_dir="./bank_remove_n_test",
            verbose=True,
        )
        summary = result["summary"]
        print("移除前准确率：", summary["accuracy_before"])
        print("移除后准确率：", summary["accuracy_after"])
        print("提升百分点：", summary["accuracy_gain_pp"])
        
        sys.exit(0)
=======
>>>>>>> ceff263d8af3a4141e129a2faaede7b864565b65
    for epoch in range(train_epochs):
        model.train()
        train_loader_nums = len(dataloaders['train'].dataset)
        train_probs = np.zeros((train_loader_nums, configs.general.num_classes), dtype = np.float32)
        train_gt    = np.zeros((train_loader_nums, 1), dtype = np.float32)
        train_k  = 0
        for data in tqdm.tqdm(dataloaders['train']):
            inputs, labels = data
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))
            if configs.general.stage == 'Purification':
                out = rn_method(student_model=model, inputs=inputs, labels=labels)
                outputs = out.output
                cls_loss = loss_functions['train'](outputs, labels)
                train_loss, nuisance_loss = rn_method.combine_with_classification_loss(classification_loss=cls_loss, lambda_nuisance=configs.general.beta, output=out)
            elif configs.general.stage == 'IM':
                outputs = model(inputs)
                train_loss = loss_functions['train'](outputs, labels)
            else:
                print(f'Error !! Pls ensure configs.general.stage == [IM | Purification].')
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            outputs =outputs.reshape(outputs.shape[0], -1)           
            labels = labels.reshape(outputs.shape[0], -1)
            train_probs[train_k: train_k + outputs.shape[0], :] = outputs.cpu().detach().numpy()
            train_gt[   train_k: train_k + outputs.shape[0]] = labels.cpu().detach().numpy()
            train_k += outputs.shape[0]
 
        lr = optimizer.param_groups[0]["lr"]
        train_pred = np.argmax(train_probs, axis=1)
        if configs.general.stage == 'Purification':
            print(f"train acc:{np.sum(train_gt.squeeze() ==train_pred)/train_k}; lr: {lr}; train_loss: {train_loss}; nuisance_loss:{nuisance_loss}")
        else:
            print(f"train acc:{np.sum(train_gt.squeeze() ==train_pred)/train_k}; lr: {lr}; train_loss: {train_loss}")
        best_acc, valid_results = eval(dataloaders, model, configs, epoch, loss_functions, best_acc)

def eval(dataloaders, model, configs, epoch, loss_functions, best_acc):
    if isinstance(model, list):
        for m in model:
            m.eval()
    else:
        model.eval()
    with torch.no_grad():
        running_loss = 0
        correct = list(0. for i in range(configs.general.num_classes))
        total = list(0. for i in range(configs.general.num_classes))
        all_labels = []
        all_outputs = []
        for data in tqdm.tqdm(dataloaders['val']):
            inputs, labels = data
            if configs.cuda.use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            val_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
            correct, total = calculate_metrics_single(labels, outputs, correct, total)
            running_loss += val_loss.data.item()

        val_epoch_loss = running_loss / len(dataloaders['val'])
        val_results = calculate_metrics(configs, all_outputs, all_labels)
        val_results['epoch'] = epoch
        val_results['status'] = 'val'
        val_results['loss'] = val_epoch_loss
        current_acc = val_results['accuracy'][3]
        print(val_results)
        log_results(configs, val_results)
        
        if  (epoch+1) >= 6 and current_acc >= best_acc:
            # save model.
            print('Best acc: %s, current acc: %s. Saving best model...' %(round(best_acc, 4), round(current_acc, 4)))
            best_acc = current_acc
            torch.save(model, f'outputs/%s/%s/{configs.general.stage}_best.pt' %(configs.general.dataset_name, configs.general.save_name))
            # best model testing.
            with torch.no_grad():
                running_loss = 0
                correct = list(0. for i in range(configs.general.num_classes))
                total = list(0. for i in range(configs.general.num_classes))
                all_labels = []
                all_outputs = []
                for data in tqdm.tqdm(dataloaders['test']):
                    inputs, labels = data
                    if configs.cuda.use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                    outputs = model(inputs)
                    test_loss = calculate_loss(configs, outputs, labels, loss_functions['test'])
                    all_outputs.append(outputs.cpu())
                    all_labels.append(labels.cpu())
                    correct, total = calculate_metrics_single(labels, outputs, correct, total)
                    running_loss += test_loss.data.item()

                test_epoch_loss = running_loss / len(dataloaders['test'])
                test_results = calculate_metrics(configs, all_outputs, all_labels)
                test_results['epoch'] = epoch
                test_results['status'] = 'test'
                test_results['loss'] = test_epoch_loss
                current_acc = test_results['accuracy'][3]
                print(test_results)
                log_results(configs, test_results)
    return best_acc, val_results

if __name__ == '__main__':
    main()

