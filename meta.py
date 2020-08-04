import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from    learner import Learner
from    copy import deepcopy
from sklearn.metrics import confusion_matrix


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """
        :param args
        :param config
        """
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt_train = args.k_spt_train
        self.k_qry_train = args.k_qry_train
        self.k_spt_test = args.k_spt_test
        self.k_qry_test = args.k_qry_test
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)
        return total_norm/counter

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)

                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q
                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct     
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)
        return loss_q, accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry, update_step_test):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4
        device = torch.device('cuda')
        querysz = x_qry.size(0)
        unk_silence_idx = torch.cat(((y_qry == 10).nonzero(), (y_qry == 11).nonzero()), 0).view(-1).cpu().numpy()
        not_unk_silence_idx = torch.from_numpy(np.delete(np.arange(querysz), unk_silence_idx)).to(device)
        not_unk_silence_idx = not_unk_silence_idx.type(torch.LongTensor)
        corrects = [0 for _ in range(update_step_test + 1)]
        unk_tps = [0 for _ in range(update_step_test + 1)]  # true positive value of the unknown class
        unk_fps = [0 for _ in range(update_step_test + 1)]  # false positive value of the unnown class
        silence_tps = [0 for _ in range(update_step_test + 1)]  # true positive value of the silence class
        silence_fps = [0 for _ in range(update_step_test + 1)]  # false positive value of the silence class
        corrects_normal = [0 for _ in range(update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), dropout_training=False, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

            unk_idx = (y_qry == self.n_way).nonzero().view(-1)  # set the unknown class idx to self.n_way after n_way keywords
            unk_tp = (pred_q[unk_idx] == self.n_way).sum().item()
            unk_tps[0] = unk_tps[0] + unk_tp

            silence_idx = (y_qry == (self.n_way+1)).nonzero().view(-1)  # set the silence class idx to self.n_way+1 after the unknown class
            silence_tp = (pred_q[silence_idx] == (self.n_way+1)).sum().item()
            silence_tps[0] = silence_tps[0] + silence_tp

            unk_fp = (pred_q[not_unk_silence_idx] == self.n_way).sum().item()
            unk_fps[0] = unk_fps[0] + unk_fp
            silence_fp = (pred_q[not_unk_silence_idx] == (self.n_way+1)).sum().item()
            silence_fps[0] = silence_fps[0] + silence_fp
            correct_normal = torch.eq(pred_q[not_unk_silence_idx], y_qry[not_unk_silence_idx]).sum().item()
            corrects_normal[0] = corrects_normal[0] + correct_normal

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, dropout_training=False, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

            unk_idx = (y_qry == self.n_way).nonzero().view(-1)
            unk_tp = (pred_q[unk_idx] == self.n_way).sum().item()
            unk_tps[1] = unk_tps[1] + unk_tp

            silence_idx = (y_qry == (self.n_way+1)).nonzero().view(-1)
            silence_tp = (pred_q[silence_idx] == (self.n_way+1)).sum().item()
            silence_tps[1] = silence_tps[1] + silence_tp

            #not_unk_silence_idx = torch.cat(((y_qry != self.n_way).nonzero(), (y_qry != (self.n_way+1)).nonzero()), 0).view(-1)
            unk_fp = (pred_q[not_unk_silence_idx] == self.n_way).sum().item()
            unk_fps[1] = unk_fps[1] + unk_fp
            silence_fp = (pred_q[not_unk_silence_idx] == (self.n_way+1)).sum().item()
            silence_fps[1] = silence_fps[1] + silence_fp
            correct_normal = torch.eq(pred_q[not_unk_silence_idx], y_qry[not_unk_silence_idx]).sum().item()
            corrects_normal[1] = corrects_normal[1] + correct_normal

        for k in range(1, update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, dropout_training=False, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k+1] = corrects[k+1] + correct

                unk_idx = (y_qry == self.n_way).nonzero().view(-1)
                unk_tp = (pred_q[unk_idx] == self.n_way).sum().item()
                unk_tps[k+1] = unk_tps[k+1] + unk_tp

                silence_idx = (y_qry == (self.n_way+1)).nonzero().view(-1)
                silence_tp = (pred_q[silence_idx] == (self.n_way+1)).sum().item()
                silence_tps[k+1] = silence_tps[k+1] + silence_tp

                #not_unk_silence_idx = torch.cat(((y_qry != self.n_way).nonzero(), (y_qry != (self.n_way+1)).nonzero()), 0).view(-1)
                unk_fp = (pred_q[not_unk_silence_idx] == self.n_way).sum().item()
                unk_fps[k+1] = unk_fps[k+1] + unk_fp
                silence_fp = (pred_q[not_unk_silence_idx] == (self.n_way+1)).sum().item()
                silence_fps[k+1] = silence_fps[k+1] + silence_fp
                correct_normal = torch.eq(pred_q[not_unk_silence_idx], y_qry[not_unk_silence_idx]).sum().item()
                corrects_normal[k+1] = corrects_normal[k+1] + correct_normal

        del net
        return corrects, unk_tps, unk_fps, silence_tps, silence_fps, corrects_normal

def main():
    pass

if __name__ == '__main__':
    main()
