# trainer.py: Utility function to train HED model
# Author: Nishanth Koganti
# Date: 2017/10/20

# Source: https://github.com/xlliu7/hed.pytorch/blob/master/trainer.py

# Issues:
# 

# import libraries
import math
import re
import os, time
import numpy as np
from PIL import Image
import os.path as osp
import glob

# import torch modules
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from visualize import Visualizations
from tensorboard_logger import configure, log_value

# utility class for training HED model
class Trainer(object):
    # init function for class
    def __init__(self, generator, optimizerG, trainDataloader, valDataloader,
                 nBatch=10, out='train', maxEpochs=1, cuda=True, gpuID=0,
                 lrDecayEpochs={}):

        # set the GPU flag
        self.cuda = cuda
        self.gpuID = gpuID
        
        # define an optimizer
        self.optimG = optimizerG
        
        # set the network
        self.generator = generator
        
        # set the data loaders
        self.valDataloader = valDataloader
        self.trainDataloader = trainDataloader
        
        # set output directory
        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)
                
        # set training parameters
        self.epoch = 0
        self.nBatch = nBatch
        self.nepochs = maxEpochs
        self.lrDecayEpochs = lrDecayEpochs
        
        self.gamma = 0.1
        self.valInterval = 500
        self.dispInterval = 100
        self.timeformat = '%Y-%m-%d %H:%M:%S'

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)
        
        
    def loss(self, data_arr, target):
        loss_list = []
        for d in data_arr:
            # compute loss for batch
            l = self.bce2d(d, target)        
            if torch.isnan(l):
                raise ValueError('loss is nan while training')
            
            loss_list.append(l)
        return loss_list
            
            
    def train(self, loadRecent=True, glyphTrain=False):
#         vis = Visualizations()
        if os.path.exists(self.out + '/log') == False:
            os.mkdir(self.out + '/log')
        configure(self.out + "/log", flush_secs=30)
        if loadRecent:
            def extract_nums_from_filename(f):
                s = re.findall(r'\d+',f)
                return (int(s[0]) if s else -1,f)
            
            checkpoints = [chkpt for chkpt in glob.glob(self.out + '/*.pth')]
            print(len(checkpoints))
            if len(checkpoints) >= 1:
                latest = self.out + '/epoch{}.pth'.format(len(checkpoints)-1)
                state = torch.load(latest)
                self.epoch= state['epoch']
                self.optimG.load_state_dict(state['optimizer'])
                self.generator.load_state_dict(state['state_dict'])
                print("[success]=> loaded checkpoint '{}' (epoch {})"
                      .format(latest, self.epoch))
                if self.epoch == self.nepochs: 
                    print('[success]=> already reached max epoch {}'.format(self.epoch))
                    return;
            else:
                print('[no checkpoints found]... starting at epoch0')
            
        # function to train network
        for epoch in range(self.epoch, self.nepochs):
            # set function to training mode
            self.generator.train()
            
            # initialize gradients
            self.optimG.zero_grad()
            
            # adjust hed learning rate
            if epoch in self.lrDecayEpochs:
                self.adjustLR()

            # train the network
            losses = []
            lossAcc = 0.0
            for i, sample in enumerate(self.trainDataloader, 0):
                # get the training batch
                data, target = sample
                
                if self.cuda:
                    data, target = data.cuda(self.gpuID), target.cuda(self.gpuID)
                data, target = Variable(data), Variable(target)
                
                # generator forward
                tar = target
                d1, d2, d3, d4, d5, d6 = self.generator(data) 
                
                loss_arr = self.loss([d1, d2, d3, d4, d5, d6], tar)
                
                # all components have equal weightage
                step_loss = sum(loss_arr)
                
                losses.append(step_loss)
                lossAcc += step_loss.item()
                
                # perform backpropogation and update network
                if i%self.nBatch == 0:
                    bLoss = sum(losses)
                
                    bLoss.backward()
                    self.optimG.step()
                    self.optimG.zero_grad()
                
                    losses = []
                    
                # visualize the loss
                if (i+1) % self.dispInterval == 0:
                    timestr = time.strftime(self.timeformat, time.localtime())
                    print("%s epoch: %d iter:%d loss:%.6f"%(timestr, epoch+1, i+1, lossAcc/self.dispInterval))
                    if not glyphTrain: 
                        log_value('Train Loss', lossAcc/self.dispInterval, i)
                    lossAcc = 0.0
                    
                # perform validation every 500 iters
                if (i+1) % self.valInterval == 0 and glyphTrain == False:
                    valLoss = self.val(epoch+1)
                    log_value('Val Loss', valLoss/self.dispInterval, i)
            # save model after every epoch
            if not glyphTrain:
                checkpoint_loc = '%s/epoch{}.pth' % (self.out)
                self.save_checkpoint({
                    'epoch': epoch+1,
                    'state_dict': self.generator.state_dict(),
                    'optimizer' : self.optimG.state_dict(),
                }, checkpoint_loc.format(epoch))
            else:
                checkpoint_loc = '%s/GLYPHDepoch{}.pth' % (self.out)
                self.save_checkpoint({
                    'epoch': epoch+1,
                    'state_dict': self.generator.state_dict(),
                    'optimizer' : self.optimG.state_dict(),
                }, checkpoint_loc.format(epoch))

    def val(self, epoch):
        # eval model on validation set
        print('Evaluation:')
        
        # convert to test mode
        self.generator.eval()
        
        # save the results
        if os.path.exists(self.out + '/images') == False:
            os.mkdir(self.out + '/images')
        dirName = '%s/images'%(self.out)
        losses = []
        lossAcc = 0.0
        # perform test inference
        for i, sample in enumerate(self.valDataloader, 0):            
            # get the test sample
            data, target = sample
            
            if self.cuda:
                data, target = data.cuda(self.gpuID), target.cuda(self.gpuID)
            data, target = Variable(data), Variable(target)
            
            # perform forward computation
            d1, d2, d3, d4, d5, d6 = self.generator.forward(data)
            
            # Compute Validation Loss
            loss_arr = self.loss([d1, d2, d3, d4, d5, d6], target)
            step_loss = sum(loss_arr)
            lossAcc += step_loss.item()
            
            # transform to grayscale images
            d1 = self.grayTrans(self.crop(d1))
            d2 = self.grayTrans(self.crop(d2))
            d3 = self.grayTrans(self.crop(d3))
            d4 = self.grayTrans(self.crop(d4))
            d5 = self.grayTrans(self.crop(d5))
            d6 = self.grayTrans(self.crop(d6))
            tar = self.grayTrans(self.crop(target))
            
            d1.save('%s/sample%d1.png' % (dirName, i))
            d2.save('%s/sample%d2.png' % (dirName, i))
            d3.save('%s/sample%d3.png' % (dirName, i))
            d4.save('%s/sample%d4.png' % (dirName, i))
            d5.save('%s/sample%d5.png' % (dirName, i))
            d6.save('%s/sample%d6.png' % (dirName, i))
            tar.save('%s/sample%dT.png' % (dirName, i))
            
        print('evaluate done')
        #Visualize here
        self.generator.train()
        return lossAcc
    
    
    # function to crop the padding pixels
    def crop(self, d):
        d_h, d_w = d.size()[2:4]
        g_h, g_w = d_h-64, d_w-64
        d1 = d[:, :, int(math.floor((d_h - g_h)/2.0)):int(math.floor((d_h - g_h)/2.0)) + g_h, int(math.floor((d_w - g_w)/2.0)):int(math.floor((d_w - g_w)/2.0)) + g_w]
        return d1
    
    def _assertNoGrad(self, variable):
        assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"

    # binary cross entropy loss in 2D
    def bce2d(self, input, target):
        n, c, h, w = input.size()
    
        # assert(max(target) == 1)
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()
        pos_index = (target_t >0)
        neg_index = (target_t ==0)
        target_trans[pos_index] = 1
        target_trans[neg_index] = 0
        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num*1.0 / sum_num
        weight[neg_index] = pos_num*1.0 / sum_num

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy(log_p, target_t, weight, reduction='mean')
        return loss

    def grayTrans(self, img):
        img = img.data.cpu().numpy()[0][0]*255.0
        img = (img).astype(np.uint8)
        img = Image.fromarray(img, 'L')
        return img

    # utility functions to set the learning rate
    def adjustLR(self):
        for param_group in self.optimG.param_groups:
            param_group['lr'] *= self.gamma 
