import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from domainbed import networks
from domainbed.lib import misc
from domainbed.algorithms import Algorithm

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float64(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

def gaussian_kernel(x1, x2, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    D = res.clamp_min_(1e-30)
    K = torch.zeros_like(D)

    for g in gamma:
        K.add_(torch.exp(D.mul(-g)))

    return K

def mmd(x, y):
    Kxx = gaussian_kernel(x, x).mean()
    Kyy = gaussian_kernel(y, y).mean()
    Kxy = gaussian_kernel(x, y).mean()
    return Kxx + Kyy - 2 * Kxy

class Myloss(nn.Module):
    def __init__(self,epsilon=1e-8):
        super(Myloss,self).__init__()
        self.epsilon = epsilon
        return
    def forward(self,input_, label, weight):
        entropy = - label * torch.log(input_ + self.epsilon) -(1 - label) * torch.log(1 - input_ + self.epsilon)
        return torch.sum(entropy * weight)/2 
    
def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ *torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def pic_loss(input_list, ad_net, coeff=None, myloss=Myloss()):
    softmax_output = input_list[0]
    focals = input_list[1].reshape(-1)
    ad_out = ad_net(softmax_output)
    ad_out = nn.Sigmoid()(ad_out)
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()

    x = softmax_output
    entropy = Entropy(x)
    entropy.register_hook(grl_hook(coeff))
    entropy = torch.exp(-entropy)
    l_dis = torch.mean(torch.abs(focals))

    source_mask = torch.ones_like(entropy)
    source_mask[softmax_output.size(0)//2:] = 0
    source_weight = entropy*source_mask
    target_mask = torch.ones_like(entropy)
    target_mask[0:softmax_output.size(0)//2] = 0
    target_weight = entropy*target_mask
    weight = source_weight / torch.sum(source_weight).detach().item() + \
             target_weight / torch.sum(target_weight).detach().item()
    l_adv = myloss(ad_out,dc_target,weight.view(-1, 1))
    return l_adv, l_dis 

class PIC(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            is_nonlinear=True)
        self.bridge = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            is_nonlinear=False)
        self.ad_net = AdversarialNetwork(num_classes, self.featurizer.n_outputs)

        self.trade_off = hparams["trade_off"]
        self.unique_filename = hparams["unique_filename"]
        self.output_dir = hparams['output_dir']

        # Optimizers
        self.optimizer = torch.optim.SGD(
            [{"params":self.featurizer.parameters(), "lr_mult":1, 'decay_mult':2}, \
            {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2},
            {"params":self.bridge.parameters(), "lr_mult":10, 'decay_mult':2},
            {"params":self.ad_net.parameters(), "lr_mult":10, 'decay_mult':2}],
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
            momentum=0.9,
            nesterov=True)

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)

    def update(self, minibatches, **kwargs):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if kwargs["path"] > 0 and kwargs["step"] == 0:
            self.refer_network = torch.load(os.path.join(self.output_dir, self.unique_filename+'_path_0.pkl'))['model']
            print("Path Information Constraint at path %d !" % (kwargs["path"]))

        all_x = [x for x, _ in minibatches]
        all_y = [y for _, y in minibatches]

        all_fea = [self.featurizer(x) for x in all_x]
        all_foc = [self.bridge(f) for f in all_fea]
        all_out = [self.classifier(f) - all_foc[i] for i,f in enumerate(all_fea)]
        classifier_loss = F.cross_entropy(torch.cat(all_out,dim=0), torch.cat(all_y,dim=0))

        if kwargs["path"] == 0 :
            refer_loss = torch.tensor(0.0)
        else:
            self.refer_network.eval()
            ref_fea = [self.refer_network.featurizer(x) for x in all_x]
            ref_foc = [self.refer_network.bridge(f) for f in ref_fea]
            ref_out = [self.refer_network.classifier(f) - ref_foc[i] for i,f in enumerate(ref_fea)]

            r_mean = sum(ref_out)/kwargs["n_train_envs"]
            s_mean = sum(all_out)/kwargs["n_train_envs"]
            refer_loss = mmd(s_mean, r_mean)

        outputs_12 = torch.cat((all_out[0], all_out[1]), dim=0)
        focals_12 = torch.cat((all_foc[0], all_foc[1]),dim=0)
        softmax_out_12 = nn.Softmax(dim=1)(outputs_12)
        l_dav, l_dis = pic_loss([softmax_out_12,focals_12], self.ad_net, calc_coeff(kwargs["step"])) 

        for i in range(1, kwargs["n_train_envs"]-1):
            tem_outputs_ind = sum(all_out[:i+1])/(i+1)
            tem_focal_ind = sum(all_foc[:i+1])/(i+1)
            outputs_ind = torch.cat((tem_outputs_ind, all_out[i+1]), dim=0)
            focal_ind = torch.cat((tem_focal_ind, all_foc[i+1]), dim=0)
            softmax_out_ind = nn.Softmax(dim=1)(outputs_ind)
  
            tem_dav, tem_dis = pic_loss([softmax_out_ind,focal_ind], self.ad_net, calc_coeff(kwargs["step"]))
            l_dav = l_dav + tem_dav
            l_dis = l_dis + tem_dis  

        l_dav = l_dav / (kwargs["n_train_envs"]-1)
        l_dis = l_dis / (kwargs["n_train_envs"]-1)

        total_loss = l_dav + classifier_loss + l_dis + self.trade_off*refer_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return {'loss_total': total_loss.item(), 'loss_cls': classifier_loss.item(), 'loss_refer': refer_loss.item()}
   
    def predict(self, x):
        feature = self.featurizer(x)
        focal = self.bridge(feature)
        out_put = self.classifier(feature)
        out_put = out_put - focal
        return out_put

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.dropout3 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    return y

  def output_num(self):
    return 1

