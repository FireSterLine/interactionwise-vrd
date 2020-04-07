import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
import numpy as np
import torchvision

ROIPool = torchvision.ops.RoIPool

class FC(nn.Module):
  """ Wrapper for linear layers with relu """

  def __init__(self, in_features, out_features, relu=True):
    super(FC, self).__init__()
    self.fc = nn.Linear(in_features, out_features)
    self.relu = nn.ReLU(inplace=True) if relu else None

  def forward(self, x):
    x = self.fc(x)
    if self.relu is not None:
      x = self.relu(x)
    return x

class Conv2d(nn.Module):
  """ Wrapper for convolutional layers with relu and batch norm """

  def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
    super(Conv2d, self).__init__()
    padding = int((kernel_size - 1) / 2) if same_padding else 0

    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
    self.bn   = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
    self.relu = nn.ReLU(inplace=True) if relu else None

  def forward(self, x):
    x = self.conv(x)
    if self.bn is not None:
      x = self.bn(x)
    if self.relu is not None:
      x = self.relu(x)
    return x

class SemSim(nn.Module):
  """ This layer computes the the similarity of an input vector with the embeddings in an embedding space.
        the similarity is given in terms of a probability distribution """

  def __init__(self, emb, mode=1):
    super(SemSim, self).__init__()
    self.emb = torch.as_tensor(emb).to("cuda:0") # TODO fix
    self.mode = mode
    if mode == 2:
      self.sig = nn.Sigmoid()
    #print(emb.shape)
  def forward(self, x):
    #print(x.shape)
    batch_size = x.shape[0]
    rel_size   = x.shape[1]
    #similarities = torch.as_tensor([[F.cosine_similarity(x[b][r], self.emb,
    #    dim=-1) for r in rel_size] for b in batch_size])
    # similarities = torch.as_tensor([[F.cosine_similarity(x[b][r], self.emb dim=-1) for r in rel_size] for b in batch_size])
    # TODO: allow batching
    #a = torch.stack([F.cosine_similarity(x[0][r], self.emb, dim=-1) for r in range(rel_size)]).to(device=utils.device)
    x = x[0]
    #print(x.device)
    #print(self.emb.device)
    if self.mode == 2:
      x = (self.sig(x)*2)-1
    cos_sim = lambda x : F.cosine_similarity(x, self.emb, dim=-1)
    shift = lambda x : x-x.min()
    scale = lambda x : x/x.sum()
    new_tens = [cos_sim(x[r]) for r in range(rel_size)]
    #new_tens = ([scale(shift(cos_sim(x[r]))) for r in range(rel_size)])
    a = torch.stack(new_tens)
    a = a.unsqueeze(0)
    #a.shape
    return a

# This function is the batched version of torch.index_select
# Source: https://discuss.pytorch.org/t/batch-index-select/62621/4
# def batched_index_select(A, indices):
#   dummy = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), A.size(2))
#   return torch.gather(A, 1, dummy) ...
# def batched_index_select(input, dim, index):
# 	views = [input.shape[0]] + \
# 		[1 if i != dim else -1 for i in range(1, len(input.shape))]
# 	expanse = list(input.shape)
# 	expanse[0] = -1
# 	expanse[dim] = -1
# 	index = index.view(views).expand(expanse)
# 	return torch.gather(input, dim, index)

def batched_index_select(A, indices): # dim, indices):
  ...
  #assert dim==0, "Warning! batched_index_select with dim = {} != 0 is untested!! I don't know if it works".format(dim)
  print()
  print("A.shape", A[0].shape)
  print("indices.shape", indices.shape)
  print("indices", indices)
  for a,index in zip(A,indices):
    print(torch.index_select(a, 0, index).shape)
    input()
  return torch.cat([ torch.index_select(a, dim, i).unsqueeze(0) for a, i in zip(A, ind) ])

# def batched_index_select(input, dim, index):
#     for ii in range(1, len(input.shape)):
#         if ii != dim:
#             index = index.unsqueeze(ii)
#     expanse = list(input.shape)
#     expanse[0] = -1
#     expanse[dim] = -1
#     index = index.expand(expanse)
#     return torch.gather(input, dim, index)


def set_trainability(model_or_params, requires_grad):
  """ Set trainability of params (or a model's params) """
  if hasattr(model_or_params, "parameters") and callable(model_or_params.parameters):
    params = model_or_params.parameters()
  else:
    params = model_or_params

  for param in params:
    param.requires_grad = requires_grad



"""




def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def load_pretrained_RO_npy(faster_rcnn_model, fname):
    params = np.load(fname).item()
    # vgg16
    vgg16_dict = faster_rcnn_model.state_dict()
    for name, val in vgg16_dict.items():
        if name.find('bn.') >= 0 or not 'conv' in name or 'lo' in name:
            continue
        i, j = int(name[4]), int(name[6]) + 1
        ptype = 'weights' if name[-1] == 't' else 'biases'
        key = 'conv{}_{}'.format(i, j)
        param = torch.from_numpy(params[key][ptype])

        if ptype == 'weights':
            param = param.permute(3, 2, 0, 1)

        val.copy_(param)

    # fc6 fc7
    frcnn_dict = faster_rcnn_model.state_dict()
    pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7', 'fc6_obj.fc': 'fc6', 'fc7_obj.fc': 'fc7'}
    for k, v in pairs.items():
        key = '{}.weight'.format(k)
        param = torch.from_numpy(params[v]['weights']).permute(1, 0)
        frcnn_dict[key].copy_(param)

        key = '{}.bias'.format(k)
        param = torch.from_numpy(params[v]['biases'])
        frcnn_dict[key].copy_(param)


def pretrain_with_det(net, det_model_path):
    det_model = torch.load(det_model_path)
    for k in det_model.keys():
        if('rpn' in k or 'bbox' in k):
            del det_model[k]

    target_keys = []
    for k in net.state_dict().keys():
        if('conv' in k):
            target_keys.append(k)

    for ix, k in enumerate(det_model.keys()):
        if('features' in k):
            det_model[target_keys[ix]] = det_model[k]
            del det_model[k]

    det_model['fc6.fc.weight'] = det_model['vgg.classifier.0.weight']
    det_model['fc6.fc.bias'] = det_model['vgg.classifier.0.bias']
    det_model['fc7.fc.weight'] = det_model['vgg.classifier.3.weight']
    det_model['fc7.fc.bias'] = det_model['vgg.classifier.3.bias']
    det_model['fc_obj.fc.weight'] = det_model['cls_score_net.weight']
    det_model['fc_obj.fc.bias'] = det_model['cls_score_net.bias']

    for k in det_model.keys():
        if('vgg' in k):
            del det_model[k]
    del det_model['cls_score_net.weight']
    del det_model['cls_score_net.bias']

    model_dict = net.state_dict()
    model_dict.update(det_model)
    net.load_state_dict(model_dict)

def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v





def clip_gradient(model, clip_norm):
    "#""Computes a gradient clipping coefficient based on gradient norm.""#"
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)

"""
