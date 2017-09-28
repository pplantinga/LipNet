import tensorflow as tf
from lipnet_model import Lipnet_Model
from tensorflow.python.client import timeline

testSpeakers = [1, 2, 20, 22]
trainSpeakers = [i for i in range(1,35) if i not in testSpeakers and i != 21 and i != 8]
dataDir = "../GRID/data"
labelDir = "../GRID/trans"

m = Lipnet_Model(trainSpeakers, testSpeakers, dataDir, labelDir)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    for epoch in range(500):
        count = 0
        for batch in m.batchify():
            sess.run(m.train_op, batch)

        m.single_word_prob *= 0.1

        cer = 0
        count = 0
        for batch in m.batchify(False):
            if cer == 0:
                print([m.index2char[i] for i in sess.run(m.a, batch)])
                print([m.index2char[i] for i in sess.run(m.b, batch)])
            cer += sess.run(m.cer, batch)
            count += 1

        cer /= count

        print("EPOCH ", epoch + 1)
        print("CER: ", cer)
