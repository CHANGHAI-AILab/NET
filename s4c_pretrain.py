# 3D_resnet based on Tencent MedicalNet
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
import os
import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
from torch import optim
from torch.utils.data import DataLoader
import time
import logging
from torch.optim import lr_scheduler
import sys
import math
import random
 
# Settings for training
root_dir = '/data/zwt/pancreas_classification/cls_with_3DUNet_seg_cls_timchen'  # type=str, help='Root directory path of data'
img_list = 'train_path_list.txt'  # type=str, help='Path for image list file'
num_seg_classes = 2 #type=int, help="Number of segmentation classes"
learning_rate = 0.001  # set to 0.001 when finetune, type=float, help= 'Initial learning rate (divided by 10 while training by lr scheduler)'
num_workers = 0 # type=int, help='Number of jobs'
batch_size = 1 # type=int, help='Batch Size'
phase = 'train' # type=str, help='Phase of train or test'
save_intervals = 10 # type=int, help='Interation for saving model'
total_epochs = 20 # type=int, help='Number of total epochs to run'
input_D = 56 # type=int, help='Input size of depth'
input_H = 448 # type=int, help='Input size of height'
input_W = 448 # type=int, help='Input size of width'
#resume_path = '' # type=str, help='Path for resume model.'
#pretrain_path = 'pretrain/resnet_50.pth' # type=str, help='Path for pretrained model.'
pretrain_path=None
new_layer_names = ['conv_seg']
#default=['upsample1', 'cmp_layer3', 'upsample2', 'cmp_layer2', 'upsample3', 'cmp_layer1', 'upsample4', 'cmp_conv1', 'conv_seg'],
# type=list, help='New layer except for backbone'
no_cuda = False # help='If true, cuda is not used.'
gpu_id = 0 # type=int, help='Gpu id lists'
basemodel = 'resnet' # type=str,help='(resnet | preresnet | wideresnet | resnext | densenet)'
model_depth = 50 # type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)'
resnet_shortcut = 'B' # type=str, help='Shortcut type of resnet (A | B)'
manual_seed = 1 # type=int, help='Manually set random seed'
ci_test = False # help='If true, ci testing is used.'
save_folder = "./trails/models/{}_{}".format(basemodel, model_depth)
 
# 3Dresnet_model backbone
#__all__ = ['ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152', 'resnet200']
def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)
def downsample_basic_block(x, planes, stride, no_cuda=no_cuda):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()
 
    out = Variable(torch.cat([out.data, zero_pads], dim=1))
 
    return out
class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out
class ResNet(nn.Module):
 
    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
 
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)
 
        self.conv_seg = nn.Sequential(
                                        nn.ConvTranspose3d(
                                        512 * block.expansion,
                                        32,
                                        2,
                                        stride=2
                                        ),
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(
                                        32,
                                        32,
                                        kernel_size=3,
                                        stride=(1, 1, 1),
                                        padding=(1, 1, 1),
                                        bias=False),
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(
                                        32,
                                        num_seg_classes,
                                        kernel_size=1,
                                        stride=(1, 1, 1),
                                        bias=False)
                                        )
 
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
 
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))
 
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_seg(x)
 
        return x
def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model
def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model
def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model
def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model
def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
 
# get 3Dresnet_model
def generate_model(basemodel, model_depth, input_D, input_H, input_W, num_seg_classes, no_cuda, phase, pretrain_path):
    assert basemodel in [
        'resnet'
    ]
 
    if basemodel == 'resnet':
        assert model_depth in [10, 18, 34, 50, 101, 152, 200]
 
        if model_depth == 10:
            model = resnet10(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=num_seg_classes)
        elif model_depth == 18:
            model = resnet18(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=num_seg_classes)
        elif model_depth == 34:
            model = resnet34(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=num_seg_classes)
        elif model_depth == 50:
            model = resnet50(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=num_seg_classes)
        elif model_depth == 101:
            model = resnet101(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=num_seg_classes)
        elif model_depth == 152:
            model = resnet152(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=num_seg_classes)
        elif model_depth == 200:
            model = resnet200(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=num_seg_classes)
 
    if not no_cuda:
        if gpu_id > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=gpu_id)
            net_dict = model.state_dict()
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()
 
    # load pretrain
    if phase != 'test' and pretrain_path:
        print ('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
 
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)
 
        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break
 
        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters,
                      'new_parameters': new_parameters}
 
        return model, parameters
 
    return model, model.parameters()
 
# define Dataset for training
class Dataset(Dataset):
 
    def __init__(self, root_dir, img_list, input_D, input_H, input_W, phase):
        with open(img_list, 'r') as f:
            self.img_list = [line.strip() for line in f]
        print("Processing {} datas".format(len(self.img_list)))
        self.root_dir = root_dir
        self.input_D = input_D
        self.input_H = input_H
        self.input_W = input_W
        self.phase = phase
 
    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")
 
        return new_data
 
    def __len__(self):
        return len(self.img_list)
 
    def __getitem__(self, idx):
 
        if self.phase == "train":
            # read image and labels
            ith_info = self.img_list[idx].split(" ")
            img_name = os.path.join(self.root_dir, ith_info[0])
            label_name = os.path.join(self.root_dir, ith_info[1])
            assert os.path.isfile(img_name)
            assert os.path.isfile(label_name)
            img = nibabel.load(img_name)  # We have transposed the data from WHD format to DHW
            assert img is not None
            mask = nibabel.load(label_name)
            assert mask is not None
 
            # data processing
            img_array, mask_array = self.__training_data_process__(img, mask)
 
            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            mask_array = self.__nii2tensorarray__(mask_array)
 
            assert img_array.shape ==  mask_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_array.shape, mask_array.shape)
            return img_array, mask_array
 
        elif self.phase == "test":
            # read image
            ith_info = self.img_list[idx].split(" ")
            img_name = os.path.join(self.root_dir, ith_info[0])
            print(img_name)
            assert os.path.isfile(img_name)
            img = nibabel.load(img_name)
            assert img is not None
 
            # data processing
            img_array = self.__testing_data_process__(img)
 
            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
 
            return img_array
 
 
    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)
 
        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
 
        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]
 
 
    def __random_center_crop__(self, data, label):
        from random import random
        """
        Random crop
        """
        target_indexs = np.where(label>0)
        [img_d, img_h, img_w] = data.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        Z_min = int((min_D - target_depth*1.0/2) * random())
        Y_min = int((min_H - target_height*1.0/2) * random())
        X_min = int((min_W - target_width*1.0/2) * random())
 
        Z_max = int(img_d - ((img_d - (max_D + target_depth*1.0/2)) * random()))
        Y_max = int(img_h - ((img_h - (max_H + target_height*1.0/2)) * random()))
        X_max = int(img_w - ((img_w - (max_W + target_width*1.0/2)) * random()))
 
        Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])
 
        Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])
 
        Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)
 
        Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)
 
        return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], label[Z_min: Z_max, Y_min: Y_max, X_min: X_max]
 
 
 
    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
 
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out
 
    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]
        data = ndimage.zoom(data, scale, order=0)
 
        return data
 
 
    def __crop_data__(self, data, label):
        """
        Random crop with different methods:
        """
        # random center crop
        data, label = self.__random_center_crop__ (data, label)
 
        return data, label
 
    def __training_data_process__(self, data, label):
        # crop data according net input size
        data = data.get_fdata()
        label = label.get_fdata()
 
        # drop out the invalid range
        data, label = self.__drop_invalid_range__(data, label)
 
        # crop data
        data, label = self.__crop_data__(data, label)
 
        # resize data
        data = self.__resize_data__(data)
        label = self.__resize_data__(label)
 
        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)
 
        return data, label
 
 
    def __testing_data_process__(self, data):
        # crop data according net input size
        data = data.get_fdata()
 
        # resize data
        data = self.__resize_data__(data)
 
        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)
 
        return data
 
# define logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
log = logging.getLogger()
 
# get model
torch.manual_seed(manual_seed)
model, parameters = generate_model(basemodel, model_depth, input_D, input_H, input_W, num_seg_classes, no_cuda, phase, pretrain_path)
# get training dataset
training_dataset = Dataset(root_dir=root_dir, img_list=img_list, input_D=input_D, input_H=input_H, input_W=input_W, phase=phase)
# get data loader
data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
# optimizer
params = [
        { 'params': parameters['base_parameters'], 'lr': learning_rate },
        { 'params': parameters['new_parameters'], 'lr': learning_rate*100 }
        ]
optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
# train from resume
# if resume_path:
#     if os.path.isfile(resume_path):
#         print("=> loading checkpoint '{}'".format(resume_path))
#         checkpoint = torch.load(resume_path)
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         print("=> loaded checkpoint '{}' (epoch {})"
#           .format(resume_path, checkpoint['epoch']))
 
# define train
def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, no_cuda):
    # settings
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    loss_seg = nn.CrossEntropyLoss(ignore_index=-1)
 
    if not no_cuda:
        loss_seg = loss_seg.cuda()
 
    model.train()
    train_time_sp = time.time()
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
 
        scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))
 
        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, label_masks = batch_data
 
            if not no_cuda:
                volumes = volumes.cuda()
 
            optimizer.zero_grad()
            out_masks = model(volumes)
            # resize label
            [n, _, d, h, w] = out_masks.shape
            new_label_masks = np.zeros([n, d, h, w])
            for label_id in range(n):
                label_mask = label_masks[label_id]
                [ori_c, ori_d, ori_h, ori_w] = label_mask.shape
                label_mask = np.reshape(label_mask, [ori_d, ori_h, ori_w])
                scale = [d*1.0/ori_d, h*1.0/ori_h, w*1.0/ori_w]
                label_mask = ndimage.zoom(label_mask, scale, order=0)
                new_label_masks[label_id] = label_mask
 
            new_label_masks = torch.tensor(new_label_masks).to(torch.int64)
            if not no_cuda:
                new_label_masks = new_label_masks.cuda()
 
            # calculating loss
            loss_value_seg = loss_seg(out_masks, new_label_masks)
            loss = loss_value_seg
            loss.backward()
            optimizer.step()
 
            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                    'Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}, avg_batch_time = {:.3f}'\
                    .format(epoch, batch_id, batch_id_sp, loss.item(), loss_value_seg.item(), avg_batch_time))
 
            # save model
            if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
            #if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                model_save_dir = os.path.dirname(model_save_path)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
 
                log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
                torch.save({
                            'ecpoch': epoch,
                            'batch_id': batch_id,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                            model_save_path)
    print('Finished training')
 
# training
train(data_loader=data_loader, model=model, optimizer=optimizer, scheduler=scheduler, total_epochs=total_epochs, save_interval=save_intervals, save_folder=save_folder, no_cuda=no_cuda)
 
 
# settting for test
phase = 'test'
resume_path = 'trails/models/resnet_50_epoch_110_batch_0.pth.tar'
img_list = './data/val.txt'
 
# read val files
def load_lines(file_path):
    """Read file into a list of lines.
    Input
      file_path: file path
    Output
      lines: an array of lines
    """
    with open(file_path, 'r') as fio:
        lines = fio.read().splitlines()
    return lines
 
# calculate the dice between prediction and ground truth
def seg_eval(pred, label, clss):
    """
    input:
        pred: predicted mask
        label: groud truth
        clss: eg. [0, 1] for binary class
    """
    Ncls = len(clss)
    dices = np.zeros(Ncls)
    [depth, height, width] = pred.shape
    for idx, cls in enumerate(clss):
        # binary map
        pred_cls = np.zeros([depth, height, width])
        pred_cls[np.where(pred == cls)] = 1
        label_cls = np.zeros([depth, height, width])
        label_cls[np.where(label == cls)] = 1
 
        # cal the inter & conv
        s = pred_cls + label_cls
        inter = len(np.where(s >= 2)[0])
        conv = len(np.where(s >= 1)[0]) + inter
        try:
            dice = 2.0 * inter / conv
        except:
            print("conv is zeros when dice = 2.0 * inter / conv")
            dice = -1
 
        dices[idx] = dice
 
    return dices
 
# define test
def test(data_loader, model, img_names, no_cuda):
    masks = []
    model.eval() # for testing
    for batch_id, batch_data in enumerate(data_loader):
        # forward
        volume = batch_data
        if not no_cuda:
            volume = volume.cuda()
        with torch.no_grad():
            probs = model(volume)
            probs = F.softmax(probs, dim=1)
 
        # resize mask to original size
        [batchsize, _, mask_d, mask_h, mask_w] = probs.shape
        data = nibabel.load(os.path.join(root_dir, img_names[batch_id]))
        data = data.get_fdata()
        [depth, height, width] = data.shape
        mask = probs[0]
        scale = [1, depth*1.0/mask_d, height*1.0/mask_h, width*1.0/mask_w]
        mask = ndimage.zoom(mask.cpu(), scale, order=1)
        mask = np.argmax(mask, axis=0)
 
        masks.append(mask)
 
    return masks
 
# getting model
checkpoint = torch.load(resume_path)
net, _ = generate_model(basemodel, model_depth, input_D, input_H, input_W, num_seg_classes, no_cuda, phase, pretrain_path)
net.load_state_dict(checkpoint['state_dict'])
 
# data tensor
testing_data = Dataset(root_dir=root_dir, img_list=img_list, input_D=input_D, input_H=input_H, input_W=input_W, phase=phase)
data_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
 
# testing
img_names = [info.split(" ")[0] for info in load_lines(img_list)]
masks = test(data_loader, net, img_names, no_cuda)
 
# evaluation: calculate dice
label_names = [info.split(" ")[1] for info in load_lines(img_list)]
Nimg = len(label_names)
dices = np.zeros([Nimg, num_seg_classes])
for idx in range(Nimg):
    label = nibabel.load(os.path.join(root_dir, label_names[idx]))
    label = label.get_fdata()
    dices[idx, :] = seg_eval(masks[idx], label, range(num_seg_classes))
 
# print result
for idx in range(1, num_seg_classes):
    mean_dice_per_task = np.mean(dices[:, idx])
    print('mean dice for class-{} is {}'.format(idx, mean_dice_per_task))