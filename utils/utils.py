from collections import Counter
import torch
import tqdm
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import torch

# WARNING: 
# There is no guarantee that it will work or be used on a model. Please do use it with caution unless you make sure everything is working.


def adjust_rho(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    rho_steps = [0.05, 0.1, 0.5, 0.5]
    if epoch <= 5:
        rho = rho_steps[0]
    elif epoch > 75:
        rho = rho_steps[3]
    elif epoch > 60:
        rho = rho_steps[2]

    else:
        rho = rho_steps[1]
    for param_group in optimizer.param_groups:
        param_group['rho'] = rho


def get_cls_num_list(image_dataset):
    c = Counter(image_dataset.img_label)
    c = sorted(c.items(), key=lambda x: x[0])  # 按类别 id 升序
    cls_num_list = []
    for i in c:
        cls_num_list.append(i[1])
    return cls_num_list

def dotproduct_similarity(A, B):
    AB = torch.mm(A, B.t())

    return AB

def forward(weights, feat):
    feat = torch.Tensor(feat)
    logits = dotproduct_similarity(feat, weights)
    
    return logits

def pnorm(weights, p):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], p)
    return ws

def get_knncentroids(dataloaders, model):

    print('===> Calculating KNN centroids.')
    torch.cuda.empty_cache()
    if isinstance(model, list):
        for m in model:
            m.eval()
    else:
        model.eval()
    feats_all, labels_all = [], []
    # Calculate initial centroids only on training data.
    with torch.set_grad_enabled(False):
        for data in tqdm.tqdm(dataloaders['train']):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            # Calculate Features of each training data
            feature_x = model.forward_features(inputs)
            feature_x = model.forward_head(feature_x, pre_logits=True)
            feats_all.append(feature_x.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    
    feats = np.concatenate(feats_all)
    labels = np.concatenate(labels_all)
    featmean = feats.mean(axis=0)
    def get_centroids(feats_, labels_):
        centroids = []        
        for i in np.unique(labels_):
            centroids.append(np.mean(feats_[labels_==i], axis=0))
        return np.stack(centroids)
    # Get unnormalized centorids
    un_centers = get_centroids(feats, labels)

    # Get l2n centorids
    l2n_feats = torch.Tensor(feats.copy())
    norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
    l2n_feats = l2n_feats / norm_l2n
    l2n_centers = get_centroids(l2n_feats.numpy(), labels)
    # Get cl2n centorids
    cl2n_feats = torch.Tensor(feats.copy())
    cl2n_feats = cl2n_feats - torch.Tensor(featmean)
    norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
    cl2n_feats = cl2n_feats / norm_cl2n
    cl2n_centers = get_centroids(cl2n_feats.numpy(), labels)
    return {'mean': featmean,
            'uncs': un_centers,
            'l2ncs': l2n_centers,   
            'cl2ncs': cl2n_centers}


@torch.no_grad()
def save_cam(t_model, inputs, labels=None, save_dir='./CAM',
             feat_layer=None, mean=None, std=None, alpha=0.5):
    """
    对一个 batch 计算 CAM 并把热力图叠加到原图保存到 save_dir。

    t_model:    分类模型（需是 conv -> GAP -> fc 结构，CAM 标准要求）
    inputs:     [B, 3, H, W]  一个 batch 的图片张量（通常已归一化）
    labels:     [B] 类别。None 则用模型预测的类来算 CAM
    save_dir:   保存目录，不存在自动创建
    feat_layer: 取特征图的卷积层模块；None 时尝试常见结构自动定位
    mean,std:   反归一化用（list/tuple，长度3）。None 则不反归一化
    alpha:      热力图叠加透明度
    """
    os.makedirs(save_dir, exist_ok=True)          # 没有就创建

    device = next(t_model.parameters()).device
    inputs = inputs.to(device)
    t_model.eval()

    # ---- 1. 用 forward hook 抓最后卷积层特征图 ----
    feats = {}
    def hook(m, i, o): feats['f'] = o
    # 自动定位特征层：优先用户指定，否则猜常见结构
    if feat_layer is None:
        if hasattr(t_model, 'layer4'):            # ResNet 系
            feat_layer = t_model.layer4
        elif hasattr(t_model, 'features'):        # VGG/DenseNet 系
            feat_layer = t_model.features
        else:
            raise ValueError("无法自动定位特征层，请手动传入 feat_layer")
    h = feat_layer.register_forward_hook(hook)

    logits = t_model(inputs)                       # [B, N_cls]
    h.remove()

    feat = feats['f']                              # [B, C, h, w] 最后卷积特征

    # ---- 2. 取分类层权重 ----
    if hasattr(t_model, 'fc'):
        fc_weight = t_model.fc.weight             # [N_cls, C]
    elif hasattr(t_model, 'classifier'):
        cls = t_model.classifier
        fc_weight = (cls[-1].weight if isinstance(cls, nn.Sequential) else cls.weight)
    else:
        raise ValueError("找不到分类层(fc/classifier)，无法用标准CAM")
    fc_weight = fc_weight.to(device)

    # ---- 3. 确定每个样本用哪个类算 CAM ----
    if labels is None:
        labels = logits.argmax(dim=1)             # 用预测类
    labels = labels.to(device)

    # ---- 4. 标准 CAM: A = sum_k w_k^c * f_k ----
    w = fc_weight[labels]                          # [B, C]
    A = torch.einsum('bc,bchw->bhw', w, feat)      # [B, h, w]
    A = F.relu(A)
    A = A / (A.amax(dim=(1, 2), keepdim=True) + 1e-8)   # 逐样本归一化[0,1]

    # 上采样到原图尺寸
    A = F.interpolate(A.unsqueeze(1), size=inputs.shape[-2:],
                      mode='bilinear', align_corners=False).squeeze(1)  # [B, H, W]

    # ---- 5. 反归一化原图用于叠加 ----
    imgs = inputs.clone()
    if mean is not None and std is not None:
        m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        s = torch.tensor(std, device=device).view(1, 3, 1, 1)
        imgs = imgs * s + m
    imgs = imgs.clamp(0, 1)

    # ---- 6. 叠加热力图并保存 ----
    A_np = A.cpu().numpy()
    imgs_np = imgs.cpu().numpy().transpose(0, 2, 3, 1)   # [B, H, W, 3]
    labels_np = labels.cpu().numpy()

    for i in range(inputs.size(0)):
        # 用 jet colormap 把 CAM 变成 RGB 热力图
        heat = cm.jet(A_np[i])[..., :3]                  # [H, W, 3]
        overlay = (1 - alpha) * imgs_np[i] + alpha * heat
        overlay = np.clip(overlay, 0, 1)

        out = (overlay * 255).astype(np.uint8)
        Image.fromarray(out).save(
            os.path.join(save_dir, f'cam_{i:03d}_cls{labels_np[i]}.png'))

    print(f"已保存 {inputs.size(0)} 张 CAM 图到 {save_dir}")
    return A   # 也返回 A，方便你后续用