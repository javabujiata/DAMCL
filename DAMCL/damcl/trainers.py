from __future__ import print_function, absolute_import
import time

from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from .utils.meters import AverageMeter
import torch
import re
from torch.autograd import Variable
# from tensorboardX import SummaryWriter
import os.path as osp
import os


class Trainer_teacher(object):
    def __init__(self, encoder, memory=None, memory_instance=None, log_loss_path=None, logging_time=None):
        super(Trainer_teacher, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.memory_instance = memory_instance
        self.log_loss_path = log_loss_path
        self.logging_time = logging_time

    def train(self, epochs, epoch, data_loader, optimizer, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        loss_class_meter = AverageMeter()
        loss_instance_meter = AverageMeter()

        end = time.time()
        writer = SummaryWriter(self.log_loss_path + "/teacher_model/" + self.logging_time[:13], comment=self.logging_time[:13])
        loop = tqdm(range(train_iters), colour='red', ncols=130)
        for i in loop:
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            #  图片描述       图片伪标签    索引
            inputs, labels, indexes = self._parse_data(inputs)

            f_out, f_out_up, f_out_down = self._forward(inputs)
            
            # compute loss with the hybrid memory
            loss_class = self.memory(f_out, f_out_up, f_out_down, labels, epoch)
            loss_instance = self.memory_instance(f_out, indexes)
            # loss_instance = loss_class
            loss = loss_class + 1.2 * loss_instance
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            loss_class_meter.update(loss_class.item())
            loss_instance_meter.update(loss_instance.item())

            batch_time.update(time.time() - end)
            end = time.time()


            loop.set_description(f"Epoch [{epoch}/{epochs - 1}]")
            loop.set_postfix(Loss=losses.avg, loss_c=loss_class_meter.avg, loss_i=loss_instance_meter.avg)
        print(re.sub(r'\x1b\[[0-9;]*m', '', str(loop)))


        writer.add_scalar("loss_total/loss", losses.avg, global_step=epoch)
        writer.add_scalar("loss_part/loss_class", loss_class_meter.avg, global_step=epoch)
        writer.add_scalar("loss_part/loss_instance", loss_instance_meter.avg, global_step=epoch)

        writer.close()


    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


class Trainer(object):
    def __init__(self, encoder, encoder_teacher, memory=None, memory_instance=None, log_loss_path=None, logging_time=None):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.encoder_teacher = encoder_teacher
        self.memory = memory
        self.memory_instance = memory_instance
        self.log_loss_path = log_loss_path
        self.logging_time = logging_time

    def train(self, epochs, epoch, data_loader, optimizer, train_iters=400):
        self.encoder.train()
        self.encoder_teacher.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        loss_class_meter = AverageMeter()
        loss_instance_meter = AverageMeter()

        end = time.time()
        writer = SummaryWriter(self.log_loss_path + "/student_model/" + self.logging_time[:13], comment=self.logging_time[:13])
        loop = tqdm(range(train_iters), colour='red', ncols=130)
        for i in loop:
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            #  图片描述       图片伪标签    索引
            inputs, labels, indexes = self._parse_data(inputs)

            f_out, f_out_up, f_out_down = self._forward(inputs)
            with torch.no_grad():
                f_out_teacher, f_out_up_teacher, f_out_down_teacher = self.encoder_teacher(inputs)

            # compute loss with the hybrid memory
            loss_class = self.memory(f_out, f_out_up, f_out_down, f_out_teacher, f_out_up_teacher, f_out_down_teacher, labels, epoch)
            loss_instance = self.memory_instance(f_out, indexes)
            # loss_instance = loss_class
            loss = loss_class + 1.2 * loss_instance
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            loss_class_meter.update(loss_class.item())
            loss_instance_meter.update(loss_instance.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()


            loop.set_description(f"Epoch [{epoch}/{epochs - 1}]")
            loop.set_postfix(Loss=losses.avg, loss_c=loss_class_meter.avg, loss_i=loss_instance_meter.avg)

        print(re.sub(r'\x1b\[[0-9;]*m', '', str(loop)))

        writer.add_scalar("loss_total/loss", losses.avg, global_step=epoch)
        writer.add_scalar("loss_part/loss_class", loss_class_meter.avg, global_step=epoch)
        writer.add_scalar("loss_part/loss_instance", loss_instance_meter.avg, global_step=epoch)

        writer.close()

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

