import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import cv2
from glob import glob
import numpy as np
from random import shuffle, random
from scipy import misc
import sys
import os.path
import os

def readVideo(filename):

    cap = cv2.VideoCapture(filename)

    frames = []
    ret, frame = cap.read()

    while ret:
        frames.append(frame)
        ret, frame = cap.read()

    cap.release()

    return np.array(frames, dtype=np.uint8)

def readVideos(dataDir, names):

    videos = []
    for name in names:
        filename = f"{dataDir}/{name}.avi"
        if os.path.isfile(filename):
            videos.append(readVideo(filename))

        else:
            print(f"Doesn't exist: {filename}")

    return videos

def readLabels(labelDir, speakers):

    labels = {}
    for speaker in speakers:
        for filename in glob(f"{labelDir}/s{speaker}_*.align"):
            name = filename[len(labelDir) + 1:-6]
            words = []
            timings = []
            for line in open(filename):
                start, stop, label = line.split()
                start = int(start) // 1000
                stop = int(stop) // 1000 + 1
                if label != "sil" and label != "sp":
                    words.append(label)
                    timings.append((start, stop))
            labels[name] = (words, timings)

    return labels

def to_sparse(tensor, lengths, maxLength):
    mask = tf.sequence_mask(lengths, maxLength)
    indices = tf.to_int64(tf.where(tf.equal(mask, True)))
    values = tf.to_int32(tf.boolean_mask(tensor, mask))
    shape = tf.to_int64(tf.shape(tensor))
    return tf.SparseTensor(indices, values, shape)


def single_word(video, label):
    
    startIndex, stopIndex = np.sort(np.random.randint(6, size=2))
    start, _ = label[1][startIndex]
    _, stop = label[1][stopIndex]
    video = video[start:stop+1]
    label = " ".join(label[0][startIndex:stopIndex+1])

    return video, label

def update_progress(progress):
    total = 30
    intprog = int(round(progress * total))
    sys.stdout.write("\r[{0}] {1:2.1f}%".format("#"*intprog + "-"*(total-intprog), progress * 100))
    sys.stdout.flush()


class Lipnet_Model():
    # Parameters
    batch_size = 50
    channel_dropout = 0.5
    #recurrent_dropout = 0.2
    hidden_size = 256
    layers = 2
    beam_width = 16
    single_word_prob = 0.9

    def __init__(self, trainSpeakers, testSpeakers, dataDir, labelDir):

        self.trainLabels = readLabels(labelDir, trainSpeakers)
        self.testLabels = readLabels(labelDir, testSpeakers)
        self.dataDir = dataDir

        self.makeStats()

        self.initialize()
        self.build_graph()

    def makeStats(self):

        self.chars = set(" ")
        self.maxVidLen = 75
        self.minVidLen = 7
        self.maxLabelLen = 0
        for label, timings in self.trainLabels.values():
            label = " ".join(label)

            # Add all characters to set, plus double characters
            prevChar = ""
            for char in label:
                self.chars.add(char)
                if char == prevChar:
                    self.chars.add(char + prevChar)
                prevChar = char

            if len(label) > self.maxLabelLen:
                self.maxLabelLen = len(label)
        
        self.char2index = {c:i for i, c in enumerate(self.chars)}
        self.index2char = {i:c for i, c in enumerate(self.chars)}

    def initialize(self):

        # Inputs
        self.videoInput = tf.placeholder(shape=(self.batch_size, self.maxVidLen, 50, 100, 3), dtype=tf.uint8)
        self.videoLengths = tf.placeholder(shape=(self.batch_size), dtype=tf.int32)
        self.targets = tf.placeholder(shape=(self.batch_size, self.maxLabelLen), dtype=tf.int32)
        self.targetLengths = tf.placeholder(shape=(self.batch_size), dtype=tf.int32)
        
        # Dropouts
        self.channel_keep_prob = tf.placeholder(dtype=tf.float32)

        # Normalize B, G, R
        inputs = tf.to_float(self.videoInput)
        mean = tf.constant([78.48, 122.29, 176.61])
        stdev = tf.constant([24.45, 28.19, 29.31])
        self.input = tf.divide(tf.subtract(inputs, mean), stdev)

    def conv3d_layer(self, inputs, filters, kernel, strides, pool_size):

        conv = tf.layers.conv3d(
                inputs      = inputs,
                filters     = filters,
                kernel_size = kernel,
                strides     = strides,
                padding     = 'same',
                activation  = tf.nn.relu)
                #data_format = 'channels_first')

        pool = tf.layers.max_pooling3d(
                inputs      = conv,
                pool_size   = pool_size,
                strides     = pool_size)
                #data_format = 'channels_first')

        # Channel-wise dropout
        drop = tf.nn.dropout(
                x           = pool,
                keep_prob   = self.channel_keep_prob,
                noise_shape = [self.batch_size, 1, 1, 1, filters])

        return drop


    def build_graph(self):
        
        # Convert to BCDHW
        #inputs = tf.transpose(self.input, perm=[0,4,1,2,3])

        conv1 = self.conv3d_layer(
                inputs    = self.input,
                filters   = 32,
                kernel    = (3,5,5),
                strides   = (1,2,2),
                pool_size = (1,2,2))

        conv2 = self.conv3d_layer(
                inputs    = conv1,
                filters   = 64,
                kernel    = (3,5,5),
                strides   = (1,1,1),
                pool_size = (1,2,2))

        conv3 = self.conv3d_layer(
                inputs    = conv2,
                filters   = 96,
                kernel    = (3,3,3),
                strides   = (1,1,1),
                pool_size = (1,2,2))
        
        # Convert to DBCHW
        #cnn_out = tf.transpose(conv3, perm=[2, 0, 1, 3, 4])

        # Prepare for RNN
        cnn_out = tf.reshape(conv3, shape=(self.batch_size, self.maxVidLen, 3 * 6 * 96))
        cnn_out = tf.transpose(cnn_out, perm=[1, 0, 2])

        # LSTM layer
        lstm_out, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw         = self.multi_cell(),
                cell_bw         = self.multi_cell(),
                inputs          = cnn_out,
                sequence_length = self.videoLengths,
                time_major      = True,
                dtype           = tf.float32)

        lstm_out = tf.concat(lstm_out, 2)

        # Output layer
        logits = tf.layers.dense(lstm_out, len(self.char2index) + 1)

        # Create train op
        sparse_targets = to_sparse(self.targets, self.targetLengths, self.maxLabelLen)
        cost = tf.reduce_mean(tf.nn.ctc_loss(
                labels = sparse_targets,
                inputs = logits,
                sequence_length = self.videoLengths,
                ignore_longer_outputs_than_inputs = True))
        self.train_op = tf.train.AdamOptimizer(0.0001).minimize(cost)

        # Create error rate op
        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, self.videoLengths, beam_width = self.beam_width)
        self.cer = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), sparse_targets))
        self.a = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values)[0]
        self.b = self.targets[0]

    def multi_cell(self):
        return rnn.MultiRNNCell([self.single_cell() for _ in range(self.layers)])

    def single_cell(self):
        #return rnn.DropoutWrapper(rnn.GRUCell(self.hidden_size), self.recurrent_keep_prob)
        return rnn.GRUCell(self.hidden_size)

    def make_fd(self, labels, names, videos, train):

        vidInput = np.zeros((self.batch_size, self.maxVidLen, 50, 100, 3), dtype=np.uint8)
        targets = np.zeros((self.batch_size, self.maxLabelLen), dtype=np.int32)
        vidLens = []
        targetLens = []

        if train:
            fd = {self.channel_keep_prob: 1. - self.channel_dropout}
            batch_size = self.batch_size // 2
        else:
            fd = {self.channel_keep_prob: 1.}
            batch_size = self.batch_size
        
        # Actual video, then left-right flip
        for j in range(self.batch_size):

            k = j % batch_size
            vid, label = videos[k], " ".join(labels[names[k]][0])

            if train:
                if random() < self.single_word_prob:
                    vid, label = single_word(videos[k], labels[names[k]])
                    if len(vid) < self.minVidLen:
                        repeat_indexes = np.random.choice(len(vid), size=self.minVidLen-len(vid))
                        indexes = np.insert(np.arange(len(vid)), repeat_indexes, repeat_indexes)
                        vid = vid[indexes]
                else:
                    # Delete and repeat 5% of frames
                    delete_indexes = np.random.choice(len(vid), size=len(vid)//20, replace=False)
                    repeat_indexes = np.random.choice(len(vid), size=len(vid)//20, replace=False)
                    indexes = np.insert(np.arange(len(vid)), repeat_indexes, repeat_indexes)
                    indexes = np.delete(indexes, delete_indexes)
                    vid = vid[indexes]

            # convert video and label to input format
            vidInput[j, :len(vid)] = vid if j < batch_size else vid[:,::-1,:]
            vidLens.append(len(vid))

            target = []
            i = 0
            while i < len(label):
                if i + 1 < len(label) and label[i + 1] == label[i]:
                    target.append(self.char2index[label[i] + label[i]])
                    i += 2
                else:
                    target.append(self.char2index[label[i]])
                    i += 1

            targets[j, :len(target)] = target
            targetLens.append(len(target))

        fd[self.videoInput] = vidInput
        fd[self.targets] = targets

        # Collect video lengths and label lengths
        fd[self.videoLengths] = vidLens
        fd[self.targetLengths] = targetLens

        return fd

    def batchify(self, train=True):

        # mirrored left-to-right
        if train:
            batch_size = self.batch_size // 2
            labels = self.trainLabels
        else:
            batch_size = self.batch_size
            labels = self.testLabels

        names = list(labels.keys())
        shuffle(names)

        for j in range(len(names) // batch_size):

            batch_names = names[j * batch_size : (j+1) * batch_size]
            batch_videos = readVideos(self.dataDir, batch_names)

            update_progress( j * batch_size / len(names) )
            yield self.make_fd(labels, batch_names, batch_videos, train)

