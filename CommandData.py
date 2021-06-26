import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
import librosa
from skimage.transform import resize
from glob import glob


class CommandData(Dataset):
    """
    put speech_commands files as :
    root :
        |- speech_commands/*/*.wav includes all audio clips
        |- train.csv
        |- test.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: meta-tasks, conain n_way * k_shot for meta-train set, n_way * n_query for meta-test set. 
          if unkown and silence classes are included, it will be n_way*k_shot+2*unk_sil_k_shot and n_way*n_query+2*unk_sil_n_query respectively.
    """

    def __init__(self, root, mode, task_type, batchsz, n_way, k_shot, k_query, k_unk_shot, k_unk_query, k_silence_shot, k_silence_query, resize, startidx=0, unk_sil_spt=False):
        """
        :param root: root path of speech commands dataset
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of cmds
        :param n_way: number of keywords in a meta-task
        :param k_shot: number of examples per keyword in the support set
        :param k_query: number of examples per keyword in the query set
        :param k_unk_shot: number of examples from the unknown class in the support set
        :param k_unk_query: number of examples from the unknown class in the query set
        :param k_silence_shot: number of examples from the silence class in the support set
        :param k_silence_query: number of examples from the silence class in the query set
        :param resize: resize to
        :param startidx: start to index label from startidx
        :param unk_sil_spt: whether to include the unknown class and silence class in the support set
        """
        self.sr = 16000
        self.batchsz = batchsz  # batch of set, not batch of cmds
        self.n_way = n_way  # n-way

        self.k_shot = k_shot  
        self.k_query = k_query  
        self.k_unk_shot = k_unk_shot
        self.k_unk_query = k_unk_query
        self.k_silence_shot = k_silence_shot
        self.k_silence_query = k_silence_query
        self.setsz = self.n_way * self.k_shot + self.k_unk_shot + self.k_silence_shot # num of samples per set
        self.querysz = self.n_way * self.k_query + self.k_unk_query + self.k_silence_query # number of samples per set for evaluation

        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        self.unk_sil_spt = unk_sil_spt
        self.background_noises = [librosa.load(x, sr=self.sr)[0] for x in glob("../data/speech_commands/_background_noise_/*.wav")]

        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (mode, batchsz, n_way, k_shot, k_query, resize))

        self.path = os.path.join(root, 'speech_commands')  # speech_commands path
        csvdata = self.loadCSV(os.path.join(root, mode + '_' + task_type + '.csv'))  # keywords path
        csvdata_unk = self.loadCSV(os.path.join(root, 'unknown_' + mode + '.csv')) # unknown class path

        self.data = []
        self.cmd2label = {}

        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[wav1, wav2, ...], [wav111, ...]]
            self.cmd2label[k] = i + self.startidx

        self.cmd2label['unknown'] = i + self.startidx + 1
        self.cmd2label['silence'] = i + self.startidx + 2
    
        for (k, v) in csvdata_unk.items():
            self.data_unk = v # [unk_wav1, unk_wav2, ...]

        self.cls_num = len(self.data)
        self.create_batch(self.batchsz)

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_cmds_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_cmds_idx)
                indexDtrain = np.array(selected_cmds_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_cmds_idx[self.k_shot:])  # idx for Dtest
                support_x.append(np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # arrange the unknown and silence classes to the support set and the query set
            selected_cmds_idx_unk = np.random.choice(len(self.data_unk), self.k_unk_shot + self.k_unk_query, False)
            np.random.shuffle(selected_cmds_idx_unk)
            selected_cmds_silence = ['silence/random_generated_silence.wav'] * (self.k_silence_shot + self.k_silence_query)
            if self.unk_sil_spt:
                indexDtrain = np.array(selected_cmds_idx_unk[:self.k_unk_shot])
                support_x.append(np.array(self.data_unk)[indexDtrain].tolist())
                support_x.append(selected_cmds_silence[:self.k_silence_shot])
            indexDtest = np.array(selected_cmds_idx_unk[self.k_unk_shot:])
            query_x.append(np.array(self.data_unk)[indexDtest].tolist())
            query_x.append(selected_cmds_silence[self.k_silence_shot:])

            # shuffle the correponding relation between support set and query set
            np.random.shuffle(support_x)
            np.random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def get_one_noise(self):
        selected_noise = self.background_noises[random.randint(0, len(self.background_noises) - 1)]
        start_idx = random.randint(0, len(selected_noise) - 1 - self.sr)
        return selected_noise[start_idx:(start_idx + self.sr)]

    def get_mix_noises(self, num_noise=1, max_ratio=0.1):
        result = np.zeros(self.sr)
        for _ in range(num_noise):
            result += random.random() * max_ratio * self.get_one_noise()
        return result / num_noise if num_noise > 0 else result

    def get_silent_wav(self, num_noise=1, max_ratio=0.1):
        return self.get_mix_noises(num_noise=num_noise, max_ratio=max_ratio)

    def get_one_word_wav(self, path, speed_rate=None):
        wav = librosa.load(path, sr=self.sr)[0]
        if speed_rate:
            wav = librosa.effects.time_stretch(wav, speed_rate)
        if len(wav) < self.sr:
            wav = np.pad(wav, (0, self.sr - len(wav)), 'constant')
        return wav[:self.sr]

    def preprocess_mfcc(self, wave):
        spectrogram = librosa.feature.melspectrogram(wave, sr=self.sr, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
        '''
        idx = [spectrogram > 0]
        spectrogram[tuple(idx)] = np.log(spectrogram[tuple(idx)])
        dct_filters = librosa.filters.dct(n_filters=40, n_input=40)
        mfcc = [np.matmul(dct_filters, x) for x in np.split(spectrogram, spectrogram.shape[1], axis=1)]
        mfcc = np.hstack(mfcc)
        mfcc = mfcc.astype(np.float32)
        '''
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram))
        return mfcc

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 1, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 1, self.resize, self.resize)
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int)
        # [querysz, 1, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 1, self.resize, self.resize)
        # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int)

        flatten_support_x = [os.path.join(self.path, item)
                             for sublist in self.support_x_batch[index] for item in sublist]
        support_y = np.array(
            [self.cmd2label[item.split('/')[0]]  # filename:down/0a2b400e_nohash_0.wav; unknown/bed/0a7c2a8d_nohash_0.wav, the command treated as label
             for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        flatten_query_x = [os.path.join(self.path, item)
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.cmd2label[item.split('/')[0]]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        unique_temp = np.unique(query_y)
        unique_list = []
        for item in list(unique_temp):
            if item != self.cmd2label['unknown'] and item != self.cmd2label['silence']:
                unique_list.append(item)

        np.random.shuffle(unique_list)
        unique_list.append(self.cmd2label['unknown'])
        unique_list.append(self.cmd2label['silence'])
        unique = np.array(unique_list)
        #print(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        for i, path in enumerate(flatten_support_x):
            if 'silence' in path:
                support_x[i] = torch.from_numpy(resize(self.preprocess_mfcc(self.get_silent_wav(num_noise=random.choice([0, 1, 2, 3]),
                max_ratio=random.choice([x / 10. for x in range(20)]))), (self.resize, self.resize), preserve_range=True)).float()
            else:
                support_x[i] = torch.from_numpy(resize(self.preprocess_mfcc(self.get_one_word_wav(path)), (self.resize, self.resize), preserve_range=True)).float()

        for i, path in enumerate(flatten_query_x):
            if 'silence' in path:
                query_x[i] = torch.from_numpy(resize(self.preprocess_mfcc(self.get_silent_wav(num_noise=random.choice([0, 1, 2, 3]),
                max_ratio=random.choice([x / 10. for x in range(20)]))), (self.resize, self.resize), preserve_range=True)).float()
            else:
                query_x[i] = torch.from_numpy(resize(self.preprocess_mfcc(self.get_one_word_wav(path)), (self.resize, self.resize), preserve_range=True)).float()

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz


if __name__ == '__main__':
    # the following episode is to view one set of images via tensorboard.
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time

    plt.ion()

    tb = SummaryWriter('runs', 'mini-imagenet')
    mini = CommandData('data', mode='train', n_way=5, k_shot=1, k_query=1, batchsz=1000, resize=168)

    for i, set_ in enumerate(mini):
        # support_x: [k_shot*n_way, 3, 84, 84]
        support_x, support_y, query_x, query_y = set_
        support_x = make_grid(support_x, nrow=2)
        query_x = make_grid(query_x, nrow=2)
        plt.figure(1)
        plt.imshow(support_x.transpose(2, 0).numpy())
        plt.pause(0.5)
        plt.figure(2)
        plt.imshow(query_x.transpose(2, 0).numpy())
        plt.pause(0.5)
        tb.add_image('support_x', support_x)
        tb.add_image('query_x', query_x)
        time.sleep(5)

    tb.close()
