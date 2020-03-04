import os
import os.path as osp
import sys
import pickle
import argparse
from tabulate import tabulate
import time

import torch
import torch.nn as nn
import torch.nn.init

from easydict import EasyDict
import globals
import utils
from lib.nets.vrd_model import vrd_model
from lib.datalayers import VRDDataLayer
from model.utils.net_utils import weights_normal_init, save_checkpoint
#, save_net, load_net, \
#      adjust_learning_rate, , clip_gradient

# from lib.datalayer import VRDDataLayer
# from lib.model import train_net, test_pre_net, test_rel_net

class vrd_trainer():

  def __init__(self, dataset_name="vg", pretrained=False):

    print("vrd_trainer() called with args:")
    print([dataset_name, pretrained])

    self.session_name = "test" # Training session name?
    self.start_epoch = 0
    self.num_epochs = 10

    self.dataset_name = dataset_name
    self.pretrained = False # TODO

    self.save_dir = osp.join(globals.models_dir, self.session_name)
    os.mkdir(self.save_dir)

    # load data layer
    print("Initializing data layer...")
    # self.datalayer = VRDDataLayer({"ds_name" : self.dataset_name, "with_bg_obj" : True}, "train")
    self.datalayer = VRDDataLayer(self.dataset_name, "train")

    self.args = EasyDict()
    self.args.n_obj   = self.datalayer.n_obj
    self.args.n_pred  = self.datalayer.n_pred

    load_pretrained = isinstance(self.pretrained, str)

    # initialize the model using the args set above
    print("Initializing VRD Model...")
    self.net = vrd_model(self.args) # TODO: load_pretrained affects how the model is initialized?
    self.net.cuda()

    # Initialize the model in some way ...
    print("Initializing weights...")
    weights_normal_init(self.net, dev=0.01)
    # Initial object embedding with word2vec
    #with open('../data/vrd/params_emb.pkl') as f:
    #    emb_init = pickle.load(f)
    #net.state_dict()['emb.weight'].copy_(torch.from_numpy(emb_init))

    print("Initializing optimizer...")
    # self.args.criterion = nn.MultiLabelMarginLoss().cuda()
    self.args.criterion = nn.MSELoss().cuda()

    self.args.lr = 0.00001
    # self.args.momentum = 0.9
    self.args.weight_decay = 0.0005

    # params = list(self.net.parameters())
    opt_params = [
      {'params': self.net.fc_spatial.parameters(),  'lr': self.args.lr},
      {'params': self.net.fc_semantic.parameters(), 'lr': self.args.lr},
      {'params': self.net.fc_fus1.parameters(),     'lr': self.args.lr},
      {'params': self.net.fc_fus2.parameters(),     'lr': self.args.lr},
    ]
    self.args.optimizer = torch.optim.Adam(opt_params,
            lr=self.args.lr,
            # momentum=self.args.momentum,
            weight_decay=self.args.weight_decay)

    # if args.resume:
    #     if osp.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         net.load_state_dict(checkpoint['state_dict'])
    #         args.optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

  def train(self):
    res_file = "output-{}.txt".format(self.session_name)

    headers = ["Epoch","Pre R@50", "ZS", "R@100", "ZS", "Rel R@50", "ZS", "R@100", "ZS"]
    res = []
    for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):

      self.__train_epoch(epoch)
      # res.append((epoch,) + test_pre_net(net, args) + test_rel_net(net, args))
      # with open(res_file, 'w') as f:
      #   f.write(tabulate(res, headers))

      save_name = osp.join(self.save_dir, "checkpoint_epoch_%d.pth.tar".format(epoch))
      save_checkpoint({
        "session": self.session_name,
        "epoch": epoch,
        "state_dict": self.net.state_dict(),
        "optimizer_state_dict": self.args.optimizer.state_dict(),
        # "loss": loss,
        # "pooling_mode": cfg.POOLING_MODE,
        # "class_agnostic": args.class_agnostic,
      }, save_name)

  def __train_epoch(self, epoch):
    self.net.train()

    # Obtain next annotation input and target
    #for spatial_features, semantic_features, target in self.datalayer:
    for i in range(10):
      # TODO: why range(10)? Loop through all of the data, maybe?

      spatial_features, semantic_features, target = self.datalayer.next()
      
      time1 = time.time()

      spatial_features  = torch.FloatTensor(spatial_features).cuda()
      semantic_features = torch.FloatTensor(semantic_features).cuda()
      target            = torch.FloatTensor(target).cuda()

      # Forward pass & Backpropagation step
      self.args.optimizer.zero_grad()
      x = self.net(spatial_features, semantic_features)

      loss = self.args.criterion(x, target)
      loss.backward()
      self.args.optimizer.step()

      time2 = time.time()
      print("TRAIN: %d, Total LOSS: %f, Time: %s".format(epoch, loss, time.strftime('%H:%M:%S', time.gmtime(int(time2 - time1)))))
      break

    """
    losses = AverageMeter()
    time1 = time.time()
    epoch_num = self.datalayer._num_instance / self.datalayer._batch_size
    for step in range(epoch_num):

      # this forward function just gets the ground truth - the annotations - for the image under consideration
      # so in reality, this forward function here is not really a network
      # the rel_so_prior here is a subset of the 100*70*70 dimensional so_prior array, which contains the predicate prob distribution for all object pairs
      # the rel_so_prior here contains the predicate probability distribution of only the object pairs in this annotation
      # Also, it might be helpful to keep in mind that this datalayer currently works for a single annotation at a time - no batching is implemented!
      image_blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, rel_labels, rel_so_prior = self.datalayer.forward()

      target = Variable(torch.from_numpy(rel_labels).type(torch.LongTensor)).cuda()
      rel_so_prior = -0.5*(rel_so_prior+1.0/self.args.num_relations)
      rel_so_prior = Variable(torch.from_numpy(rel_so_prior).type(torch.FloatTensor)).cuda()

      # backward
      # this is where the forward function of the trainable VRD network is really applied
      self.args.optimizer.zero_grad()
      obj_score, rel_score = self.net(image_blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, self.args)

      loss = self.args.criterion((rel_so_prior+rel_score).view(1, -1), target)
      losses.update(loss.data[0])
      loss.backward()
      self.args.optimizer.step()

      if step % self.args.print_freq == 0:
        time2 = time.time()
        print "TRAIN:%d, Total LOSS:%f, Time:%s" % (step, losses.avg, time.strftime('%H:%M:%S', time.gmtime(int(time2 - time1))))
        time1 = time.time()
        losses.reset()
    """

if __name__ == '__main__':
  trainer = vrd_trainer()

  trainer.train()
