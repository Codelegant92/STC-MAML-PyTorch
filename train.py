import  torch, os
import  numpy as np
from    CommandData import CommandData
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from tensorboardX import SummaryWriter
import logging
from utils import set_logger
from meta import Meta

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs, axis=0), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def train(model, mini_train, model_path, resume_itr, device, writer):

    if resume_itr > 0:
        model = torch.load(model_path+'/model-'+str(resume_itr)+'.pth')
        model.eval(model_path+'/model-'+str(resume_itr)+'.pth')

    db = DataLoader(mini_train, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

    #torch.manual_seed(1)
    #torch.cuda.manual_seed_all(1)
    #np.random.seed(1)

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        loss_q, accs = model(x_spt, y_spt, x_qry, y_qry)
        step = resume_itr+step
        writer.add_scalar('scalar/loss_q', loss_q, step)
        writer.add_scalar('scalar/training_acc', accs[-1], step)

        if step % 100 == 0:
            print('iteration:', step, '\ttraining acc:', accs[-1])
            logging.info("iteration: {} \t training acc: {}".format(step, accs[-1]))
            print('iteration:', step, '\ttraining loss:', loss_q.item())
            logging.info("iteration: {} \t training loss: {}".format(step, loss_q.item()))
        # evaluation
        if step % 500 == 0:  
            print("=====================================[iteration: {}][keywords: [#-spt-train: {}], [#-qry-train: {}], [#-spt-test: {}], [#-qry-test: {}]]\
                [unk & silence: [#-spt-train: {}], [#-qry-train: {}], [#-spt-test: {}], [#-qry-test: {}]]===============================".format(step, args.k_spt_train,
                args.k_qry_train, args.k_spt_test, args.k_qry_test, args.k_spt_unk_train, args.k_qry_unk_train, args.k_spt_unk_test, args.k_qry_unk_test))
            logging.info("=====================================[iteration: {}][keywords: [#-spt-train: {}], [#-qry-train: {}], [#-spt-test: {}], [#-qry-test: {}]]\
                [unk & silence: [#-spt-train: {}], [#-qry-train: {}], [#-spt-test: {}], [#-qry-test: {}]]===============================".format(step, args.k_spt_train,
                args.k_qry_train, args.k_spt_test, args.k_qry_test, args.k_spt_unk_train, args.k_qry_unk_train, args.k_spt_unk_test, args.k_qry_unk_test))
            # because of the limited dataset, we don't use a validation set. instead we evaluate the training model every 500 iterations.
            mini_test = CommandData('../data', mode='test', task_type=args.task_type, n_way=args.n_way, k_shot=args.k_spt_test,
                             k_query=args.k_qry_test, k_unk_shot=args.k_spt_unk_test, k_unk_query=args.k_qry_unk_test,
                             k_silence_shot=args.k_spt_silence_test, k_silence_query=args.k_qry_silence_test,
                             batchsz=100, resize=args.imgsz, unk_spt=False)
            db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
            accs_all_test = []
            accs_normal_test = []
            unk_tps_test = []
            silence_tps_test = []

            for x_spt, y_spt, x_qry, y_qry in db_test:
                x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                if step == 0:   # to get the results of purely supervised learning
                    corrects, unk_tps, unk_fps, silence_tps, silence_fps, corrects_normal = model.finetunning(x_spt, y_spt, x_qry, y_qry, 100)
                else:
                    corrects, unk_tps, unk_fps, silence_tps, silence_fps, corrects_normal = model.finetunning(x_spt, y_spt, x_qry, y_qry, args.update_step_test)
                
                num_keywords_qry_samples_test = args.n_way*args.k_qry_test*1.0
                num_unknown_qry_samples_test = args.k_qry_unk_test*1.0
                num_silence_qry_samples_test = args.k_qry_silence_test*1.0
                num_all_qry_samples_test = num_keywords_qry_samples_test + num_unknown_qry_samples_test + num_silence_qry_samples_test
                accs_all_test.append(np.array(corrects)/num_all_qry_samples_test)
                accs_normal_test.append(np.array(corrects_normal)/num_keywords_qry_samples_test)
                unk_tps_test.append(np.array(unk_tps)/num_unknown_qry_samples_test)
                silence_tps_test.append(np.array(silence_tps)/num_silence_qry_samples_test)

            # [b, update_step+1]
            #accs = np.array(accs_all_test1).mean(axis=0).astype(np.float16)
            accs = np.array(accs_all_test)
            mean_accs, h_accs = mean_confidence_interval(accs)
            print('Test acc all(m):', mean_accs)
            print('Test acc all(h):', h_accs)
            logging.info('Test acc all(m): {}'.format(mean_accs))
            logging.info('Test acc all(h): {}'.format(h_accs))

            accs_normal = np.array(accs_normal_test)
            mean_accs_normal, h_accs_normal = mean_confidence_interval(accs_normal)
            print('Test acc normal(m):', mean_accs_normal)
            print('Test acc normal(h):', h_accs_normal)
            logging.info('Test acc normal(m): {}'.format(mean_accs_normal))
            logging.info('Test acc normal(h): {}'.format(h_accs_normal))

            unk_tps = np.array(unk_tps_test)
            mean_unk_tps, h_unk_tps = mean_confidence_interval(unk_tps)
            print('Unknown true positive(m):', mean_unk_tps)
            print('Unknown true positive(h):', h_unk_tps)
            logging.info('Unknown true positive(m): {}'.format(mean_unk_tps))
            logging.info('Unknown true positive(h): {}'.format(h_unk_tps))

            silence_tps = np.array(silence_tps_test)
            mean_silence_tps, h_silence_tps = mean_confidence_interval(silence_tps)
            print('Silence true positive(m):', mean_silence_tps)
            print('Silence true positive(h):', h_silence_tps)
            logging.info('Silence true positive(m): {}'.format(mean_silence_tps))
            logging.info('Silence true positive(h): {}'.format(h_silence_tps))

        if step % 500 == 0:
            torch.save(model, model_path+'/'+'model-'+str(step)+'.pth')
    writer.close()

def test(model, model_file, device):
    print(model_file)
    model = torch.load(model_file)
    model.eval()
 
    print("=====================================[keywords: [#-spt-train: {}], [#-qry-train: {}], [#-spt-test: {}], [#-qry-test: {}]]\
        [unk & silence: [#-spt-train: {}], [#-qry-train: {}], [#-spt-test: {}], [#-qry-test: {}]]===============================".format(args.k_spt_train,
        args.k_qry_train, args.k_spt_test, args.k_qry_test, args.k_spt_unk_train, args.k_qry_unk_train, args.k_spt_unk_test, args.k_qry_unk_test))
    logging.info("=====================================[keywords: [#-spt-train: {}], [#-qry-train: {}], [#-spt-test: {}], [#-qry-test: {}]]\
        [unk & silence: [#-spt-train: {}], [#-qry-train: {}], [#-spt-test: {}], [#-qry-test: {}]]===============================".format(args.k_spt_train,
        args.k_qry_train, args.k_spt_test, args.k_qry_test, args.k_spt_unk_train, args.k_qry_unk_train, args.k_spt_unk_test, args.k_qry_unk_test))
    
    mini_test = CommandData('../data', mode='test', task_type=args.task_type, n_way=args.n_way, k_shot=args.k_spt_test,
                             k_query=args.k_qry_test, k_unk_shot=args.k_spt_unk_test, k_unk_query=args.k_qry_unk_test,
                             k_silence_shot=args.k_spt_silence_test, k_silence_query=args.k_qry_silence_test,
                             batchsz=100, resize=args.imgsz, unk_spt=args.unk_spt)

    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
    accs_all_test = []
    accs_normal_test = []
    unk_tps_test = []
    silence_tps_test = []

    for x_spt, y_spt, x_qry, y_qry in db_test:
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        #accs, unk_tps, unk_fps, silence_tps, silence_fps, accs_normal = model.finetunning(x_spt, y_spt, x_qry, y_qry, args.update_step_test)
        accs, unk_tps, unk_fps, silence_tps, silence_fps, accs_normal = model.finetunning(x_spt, y_spt, x_qry, y_qry, 1000)
        num_keywords_qry_samples_test = args.n_way*args.k_qry_test*1.0
        num_unknown_qry_samples_test = args.k_qry_unk_test*1.0
        num_silence_qry_samples_test = args.k_qry_silence_test*1.0
        num_all_qry_samples_test = num_keywords_qry_samples_test + num_unknown_qry_samples_test + num_silence_qry_samples_test
        accs_all_test.append(np.array(corrects)/num_all_qry_samples_test)
        accs_normal_test.append(np.array(corrects_normal)/num_keywords_qry_samples_test)
        unk_tps_test.append(np.array(unk_tps)/num_unknown_qry_samples_test)
        silence_tps_test.append(np.array(silence_tps)/num_silence_qry_samples_test)

    accs = np.array(accs_all_test)
    mean_accs, h_accs = mean_confidence_interval(accs)
    print('Test acc all(m):', mean_accs)
    print('Test acc all(h):', h_accs)
    logging.info('Test acc all(m): {}'.format(mean_accs))
    logging.info('Test acc all(h): {}'.format(h_accs))

    accs_normal = np.array(accs_normal_test)
    mean_accs_normal, h_accs_normal = mean_confidence_interval(accs_normal)
    print('Test acc normal(m):', mean_accs_normal)
    print('Test acc normal(h):', h_accs_normal)
    logging.info('Test acc normal(m): {}'.format(mean_accs_normal))
    logging.info('Test acc normal(h): {}'.format(h_accs_normal))

    unk_tps = np.array(unk_tps_test)
    mean_unk_tps, h_unk_tps = mean_confidence_interval(unk_tps)
    print('Unknown true positive(m):', mean_unk_tps)
    print('Unknown true positive(h):', h_unk_tps)
    logging.info('Unknown true positive(m): {}'.format(mean_unk_tps))
    logging.info('Unknown true positive(h): {}'.format(h_unk_tps))

    silence_tps = np.array(silence_tps_test)
    mean_silence_tps, h_silence_tps = mean_confidence_interval(silence_tps)
    print('Silence true positive(m):', mean_silence_tps)
    print('Silence true positive(h):', h_silence_tps)
    logging.info('Silence true positive(m): {}'.format(mean_silence_tps))
    logging.info('Silence true positive(h): {}'.format(h_silence_tps))


def main():

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    print(args)
    # model architecture configuration
    config = [
        ('conv2d', [args.num_filters, 1, 3, 3, 2, 1]),
        ('relu', [True]),
        ('bn', [args.num_filters]),
        ('conv2d', [args.num_filters, args.num_filters, 3, 3, 2, 1]),
        ('relu', [True]),
        ('bn', [args.num_filters]),
        ('conv2d', [args.num_filters, args.num_filters, 3, 3, 2, 1]),
        ('relu', [True]),
        ('bn', [args.num_filters]),
        ('conv2d', [args.num_filters, args.num_filters, 3, 3, 2, 1]),
        ('relu', [True]),
        ('bn', [args.num_filters]),
        ('flatten', []),
        ('linear', [args.n_way+2, args.num_filters*9])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total sampled meta-task number
    if args.train == 'True':
        mini_train = CommandData('../data', mode='train', task_type=args.task_type, n_way=args.n_way, k_shot=args.k_spt_train,
                        k_query=args.k_qry_train, k_unk_shot=args.k_spt_unk_train, k_unk_query=args.k_qry_unk_train,
                        k_silence_shot=args.k_spt_silence_train, k_silence_query=args.k_qry_silence_train,
                        batchsz=160000, resize=args.imgsz, unk_spt=args.unk_spt)

    exp_string = 'cls_'+str(args.n_way)+'.tskn_'+str(args.task_num)+'.spttrain_'+str(args.k_spt_train)+'.qrytrain_'+str(args.k_qry_train)+'.numstep'+str(args.update_step)+'.updatelr'+str(args.update_lr)
    model_path = args.logdir + '/' + exp_string
    model_file = None

    if args.train=='True':
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            print("logs directory ", args.logdir, " created!")
        writer = SummaryWriter(model_path)
        set_logger(os.path.join(args.logdir, 'train.log'))
        train(maml, mini_train, model_path, args.resume_itr, device, writer)
    else:
        if args.test_iter >= 0:
            model_file = model_path + '/' + 'model-' + str(args.test_iter) + '.pth'
            test(maml, model_file, device)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--task_type', type=str, help='few-shot commands recognition, or few-shot digits recognition', default='commands', choices=['commands', 'digits'])
    argparser.add_argument('--num_filters', type=int, default=64)
    argparser.add_argument('--iterations', type=int, help='iteration number', default=8000)
    argparser.add_argument('--n_way', type=int, help='n way, number of user-defined keywords', default=10)
    argparser.add_argument('--unk_sil_spt', type=bool, help='whether to feed unknown or silence examples to support set', default=False)

    argparser.add_argument('--k_spt_train', type=int, help='number of examples per selected keyword in the support set in the meta-learning stage', default=1)
    argparser.add_argument('--k_qry_train', type=int, help='number of examples per selected keyword in the query set in the meta-learning stage', default=15)
    argparser.add_argument('--k_spt_unk_train', type=int, help='number of examples from the unknown class in the support set in the meta-learning stage', default=0)
    argparser.add_argument('--k_qry_unk_train', type=int, help='number of examples from the unknown class in the query set in the meta-learning stage', default=15)
    argparser.add_argument('--k_spt_silence_train', type=int, help='number of examples from the silence class in the support set in the meta-learning stage', default=0)
    argparser.add_argument('--k_qry_silence_train', type=int, help='number of examples from the silence class in the query set in the meta-learning stage', default=15)
    argparser.add_argument('--k_spt_test', type=int, help='number of examples per user-defined keyword in the support set in the fine-tuning stage', default=1)
    argparser.add_argument('--k_qry_test', type=int, help='number of examples per user-defined keyword in the query set in the fine-tuning stage', default=100)
    argparser.add_argument('--k_spt_unk_test', type=int, help='number of examples from the unknown class in the support set in the fine-tuning stage', default=0)
    argparser.add_argument('--k_qry_unk_test', type=int, help='number of examples from the unknown class in the query set in the fine-tuning stage', default=100)
    argparser.add_argument('--k_spt_silence_test', type=int, help='number of examples from the silence class in the support set in the fine-tuning stage', default=0)
    argparser.add_argument('--k_qry_silence_test', type=int, help='number of examples from the silence class in the query set in the fine-tuning stage', default=100)

    argparser.add_argument('--imgsz', type=int, help='imgsz, resized shape', default=40)
    argparser.add_argument('--imgc', type=int, help='imgc, channel', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=16)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for fine-tunning in the fine-tuning stage', default=10)
    argparser.add_argument('--logdir', type=str, default='logs/CommandData5shot')
    argparser.add_argument('--train', type=str, default='True')
    argparser.add_argument('--resume_itr', type=int, default=0)
    argparser.add_argument('--test_iter', type=int, default=-1)
    args = argparser.parse_args()

    main()
