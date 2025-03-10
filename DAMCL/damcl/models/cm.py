import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            # ctx.features[y] = 0.1 * ctx.features[y] + (1. - 0.1) * x 
            # ctx.features[y] = x                       
            ctx.features[y] /= ctx.features[y].norm()
        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)


        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))  # Hard
            # median = np.argmax(np.array(distances))  # Easy
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            # ctx.features[y] = x                       
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_BestCluster(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())
        # outputs = F.softmax(torch.cdist(inputs.clone(), ctx.features.clone()), dim=1)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
            
        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for idx in batch_centers.keys():
            fea_idx = torch.stack(batch_centers[idx], dim=0).to(device)
            distances = torch.cdist(fea_idx, fea_idx).to("cpu")
            median = np.argmin(torch.sum(distances, dim=1).to("cpu"))
            ctx.features[idx] = ctx.momentum * ctx.features[idx] + (1. - ctx.momentum) * fea_idx[median]
            ctx.features[idx] /= ctx.features[idx].norm()

        return grad_inputs, None, None, None
    

def cm_bestcluster(inputs, indexes, features, momentum=0.5):
    return CM_BestCluster.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory_teacher(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, lambda2=None):
        super(ClusterMemory_teacher, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = 0.0
        self.temp = temp
        self.use_hard = use_hard

        self.lambda2 = lambda2

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('features_up', torch.zeros(num_samples, num_features))
        self.register_buffer('features_down', torch.zeros(num_samples, num_features))

    def forward(self, inputs, inputs_up, inputs_down, targets, epoch):

        inputs = F.normalize(inputs, dim=1).cuda()
        inputs_up = F.normalize(inputs_up, dim=1).cuda()
        inputs_down = F.normalize(inputs_down, dim=1).cuda()

        # query与代理中心求欧几里得距离再softmax
        inputs_soft = F.softmax(torch.cdist(inputs.clone(), self.features.clone()), dim=1)
        inputs_up_soft = F.softmax(torch.cdist(inputs_up.clone(), self.features_up.clone()), dim=1)
        inputs_down_soft = F.softmax(torch.cdist(inputs_down.clone(), self.features_down.clone()), dim=1)

        # 批量大小 × 伪标签类别数
        outputs = cm(inputs, targets, self.features, self.momentum)
        outputs_up = cm(inputs_up, targets, self.features_up, self.momentum)
        outputs_down = cm(inputs_down, targets, self.features_down, self.momentum)

        outputs /= self.temp
        outputs_up /= self.temp
        outputs_down /= self.temp


        loss = ((1.0-self.lambda2) * (F.cross_entropy(outputs, targets) + F.cross_entropy(inputs_soft, targets)) +
                self.lambda2 * (F.cross_entropy(outputs_up, targets) + F.cross_entropy(inputs_up_soft, targets)) +
                self.lambda2 * (F.cross_entropy(outputs_down, targets) + F.cross_entropy(inputs_down_soft, targets)))
        # loss = ((1.0-self.lambda2) * F.cross_entropy(outputs, targets) +
        #         self.lambda2 * F.cross_entropy(outputs_up, targets) +
        #         self.lambda2 * F.cross_entropy(outputs_down, targets))

        return loss


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, lambda2=None, mu=None):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = 0.0
        self.temp = temp
        self.use_hard = use_hard

        self.lambda2 = lambda2
        self.mu = mu

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('features_up', torch.zeros(num_samples, num_features))
        self.register_buffer('features_down', torch.zeros(num_samples, num_features))

    def forward(self, inputs, inputs_up, inputs_down, inputs_teacher, inputs_up_teacher, inputs_down_teacher, targets, epoch):

        inputs = F.normalize(inputs, dim=1).cuda()
        inputs_up = F.normalize(inputs_up, dim=1).cuda()
        inputs_down = F.normalize(inputs_down, dim=1).cuda()
        inputs_teacher = F.normalize(inputs_teacher, dim=1).cuda()
        inputs_up_teacher = F.normalize(inputs_up_teacher, dim=1).cuda()
        inputs_down_teacher = F.normalize(inputs_down_teacher, dim=1).cuda()


        # query与代理中心求欧几里得距离再softmax
        inputs_soft = F.softmax(torch.cdist(inputs.clone(), self.features.clone()), dim=1)
        inputs_up_soft = F.softmax(torch.cdist(inputs_up.clone(), self.features_up.clone()), dim=1)
        inputs_down_soft = F.softmax(torch.cdist(inputs_down.clone(), self.features_down.clone()), dim=1)

        if epoch == -1:
            outputs = cm(inputs, targets, self.features, 1.0)
            outputs_up = cm(inputs_up, targets, self.features_up, 1.0)
            outputs_down = cm(inputs_down, targets, self.features_down, 1.0)
        else:
            outputs = cm(inputs, targets, self.features,self.momentum)
            outputs_up = cm(inputs_up, targets, self.features_up, self.momentum)
            outputs_down = cm(inputs_down, targets, self.features_down, self.momentum)

        outputs /= self.temp
        outputs_up /= self.temp
        outputs_down /= self.temp


        loss_distill = nn.MSELoss(reduce=False)(inputs, inputs_teacher).mean(0).sum()
        loss_distill_up = nn.MSELoss(reduce=False)(inputs_up, inputs_up_teacher).mean(0).sum()
        loss_distill_down = nn.MSELoss(reduce=False)(inputs_down, inputs_down_teacher).mean(0).sum()
        loss = ((1.0-self.lambda2) * (F.cross_entropy(outputs, targets) + 
                                      self.mu * loss_distill + 
                                      F.cross_entropy(inputs_soft, targets)) +
                self.lambda2 * (F.cross_entropy(outputs_up, targets) + 
                                self.mu * loss_distill_up + 
                                F.cross_entropy(inputs_up_soft, targets)) +
                self.lambda2 * (F.cross_entropy(outputs_down, targets) + 
                                self.mu * loss_distill_down + 
                                F.cross_entropy(inputs_down_soft, targets)))

        return loss


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x                                  
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        # 动量设为0，直接拿生成特征替换记忆库特征进行更新
        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, indexes):
        inputs = hm(inputs, indexes, self.features, self.momentum)
        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            masked_exps = vec * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()
        sim = torch.zeros(labels.max()+1, B).float().cuda()
        inputs = torch.exp(inputs)
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        mask = (nums>0).float()
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        return F.nll_loss(torch.log(masked_sim+1e-6), targets)





