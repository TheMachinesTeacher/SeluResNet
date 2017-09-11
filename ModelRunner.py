#!/usr/bin/env python3
from multiprocessing import Process
from Cifar10 import Cifar10
from Train import Train

class TensorflowFuncRun(Process):

    def __init__(self, modelClass): 
        super(TensorflowFuncRun, self).__init__()
        self.modelClass = modelClass
        self.train = Train()

    def run(self):
        images_train, labels_train = self.modelClass.getInputs(False, True, True)
        images_val, labels_val = self.modelClass.getInputs(True)
        is_training = tf.placeholder('bool', [], name='is_training')

        images, labels = tf.cond(is_training, 
                lambda: (images_train, labels_train),
                lambda: (images_val, labels_val))
        logits = self.modelClass.getModel(images)
        self.train.train(is_training, logits, images, labels)

def main()
    funcsToRun = [Cifar10(True), Cifar10(False)]
    for i in range(len(funcsToRun)):
        proc = TensorflowFuncRun(funcsToRun[i])
        proc.start()
        proc.join()

if __name__ == '__main__':
    tf.app.run()
