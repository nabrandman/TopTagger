import tensorflow as tf
import numpy as np

from glob import glob

from CustomQueueRunner import CustomQueueRunner
from FileNameQueue import FileNameQueue
from DataGetter import DataGetter

class DataSample:
    def __init__(self, dataSet, nEpoch, batchSize, variables, inputDataQueue, sumScaleFactor, signal = True, background = True, domain = 0, ptReweight=False):

        self.dataSet = dataSet

        self.inputDataQueue = inputDataQueue
        self.variables = variables
        self.signal = signal
        self.background = background
        self.domain = domain
        self.ptReweight = ptReweight

        self.queue = tf.queue.FIFOQueue(capacity = 32768, shapes = inputDataQueue.shapes, dtypes = inputDataQueue.dtypes)

        #enqueue delay
        self.scaleFactor = self.dataSet.xsec*self.dataSet.rescale*self.dataSet.kFactor/self.dataSet.Nevts
        self.batchSize = batchSize

        #create file name list form file glob
        self.fileList = glob(self.dataSet.fileGlob)
        #print('\n\nDataSample\n\n')
        #print(len(self.fileList))
        #print(self.fileList)
        #print('\n\n')
        #Create file queue of input files 
        self.fileQueue = FileNameQueue(self.fileList, nEpoch)

        #create CustomRunner for this dataset 
        self.customRunner = CustomQueueRunner(self.batchSize, variables, self.fileQueue, self.queue, signal, background, domain, ptReweight, self.dataSet.weightHist, self.dataSet.include)

    def get_data(self):
        dg = DataGetter(self.variables, signal=self.signal, background=self.background, domain=self.domain, bufferData = False, weightHist=self.dataSet.weightHist, include=self.dataSet.include)

        #print(len(self.fileList))
        #print(self.fileList)
        #print('\n\n')
        #print('\n\nget_data\n\n')
        #print(self.fileList[0])
        for fileName in self.fileList:
            data = dg.importData(fileName, ptReweight=self.ptReweight)
            batch_idx = 0
            nSamples = data["data"].shape[0]
            #print()
            #print(fileName, data['data'].shape[0])
            #print()
            while (batch_idx < nSamples-1):
                batch_idx += 1
                #print(fileName, batch_idx)
                yield {'x': data["data"][batch_idx-1], 'p_': data["domain"][batch_idx-1], 'wgt': data["weights"][batch_idx-1], 'y_': data["labels"][batch_idx-1]}

        return

    def start_threads(self, sess, coord, n_threads=1):
        return self.customRunner.start_threads(sess, coord, n_threads)

    def getEnqueueOp(self, nSample, nQueue = 1):
        return [self.inputDataQueue.enqueue_many(self.queue.dequeue_many(nSample)),] * nQueue
