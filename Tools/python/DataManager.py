import numpy as np
import math
import tensorflow as tf
import threading
import queue
from time import sleep

from DataSample import DataSample

class DataManager:

    def __init__(self, variables, nEpoch, nFeatures, nLabels, nDomains, nWeigts, ptReweight, signalDataSets, backgroundDataSets):
        self.nEpoch = nEpoch

        #Define input data queue
        self.inputDataQueue = tf.compat.v1.RandomShuffleQueue(capacity=65536*2, min_after_dequeue=65536*2 - 65536/2, shapes=[[nFeatures], [nLabels], [nDomains], [nWeigts]], dtypes=[tf.float32, tf.float32, tf.float32, tf.float32])

        #Add DataSample objects for each data set used 
        self.sigScaleSum = 0
        for dataSet in signalDataSets:
            self.sigScaleSum += dataSet.xsec*dataSet.rescale*dataSet.kFactor/dataSet.Nevts

        batchSize = int(65536 / 4)
        self.sigDataSamples = []
        #print('\n\nSignal\n')
        for dataSet in signalDataSets:
            #print(dataSet.fileGlob)
            self.sigDataSamples.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.sigScaleSum, signal=True, background=False, domain=dataSet.domain, ptReweight=ptReweight))

        self.bgScaleSum = 0
        for dataSet in signalDataSets:
            self.bgScaleSum += dataSet.xsec*dataSet.rescale*dataSet.kFactor/dataSet.Nevts

        #self.bgDataSamples_one = []
        #self.bgDataSamples_two = []
        self.bgDataSamples = []
        self.bgDataSamples_mc = []
        self.bgDataSamples_tt = []
        self.bgDataSamples_cr = []
        self.bgDataSamples_15to20 = []
        self.bgDataSamples_20to30 = []
        self.bgDataSamples_30to50 = []
        self.bgDataSamples_50to80 = []
        self.bgDataSamples_80to120 = []
        self.bgDataSamples_120to170 = []
        self.bgDataSamples_170to300 = []
        self.bgDataSamples_300to470 = []
        self.bgDataSamples_470to600 = []
        self.bgDataSamples_600to800 = []
        self.bgDataSamples_800to1000 = []
        self.bgDataSamples_1000to1500 = []
        self.bgDataSamples_1500to2000 = []
        self.bgDataSamples_2000to2500 = []
        self.bgDataSamples_2500to3000 = []

        n_bgDataSamples = 0
        #print('\n\nBackground\n')
        for dataSet in backgroundDataSets:
            if '_15to20_' in dataSet.fileGlob:
                self.bgDataSamples_15to20.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            elif '_20to30_' in dataSet.fileGlob:
                self.bgDataSamples_20to30.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            elif '_30to50_' in dataSet.fileGlob:
                self.bgDataSamples_30to50.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            elif '_50to80_' in dataSet.fileGlob:
                self.bgDataSamples_50to80.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            elif '_80to120_' in dataSet.fileGlob:
                self.bgDataSamples_80to120.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            elif '_120to170_' in dataSet.fileGlob:
                self.bgDataSamples_120to170.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            elif '_170to300_' in dataSet.fileGlob:
                self.bgDataSamples_170to300.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            elif '_300to470_' in dataSet.fileGlob:
                self.bgDataSamples_300to470.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            elif '_470to600_' in dataSet.fileGlob:
                self.bgDataSamples_470to600.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            elif '_600to800_' in dataSet.fileGlob:
                self.bgDataSamples_600to800.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            elif '_800to1000_' in dataSet.fileGlob:
                self.bgDataSamples_800to1000.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            elif '_1000to1500_' in dataSet.fileGlob:
                self.bgDataSamples_1000to1500.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            elif '_1500to2000_' in dataSet.fileGlob:
                self.bgDataSamples_1500to2000.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            elif '_2000to2500_' in dataSet.fileGlob:
                self.bgDataSamples_2000to2500.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            elif '_2500to3000_' in dataSet.fileGlob:
                self.bgDataSamples_2500to3000.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            if 'TTToLNu2Q' in dataSet.fileGlob:
                self.bgDataSamples_tt.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            elif 'QCDCR' in dataSet.fileGlob:
                self.bgDataSamples_cr.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            else:
                self.bgDataSamples_mc.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))

        for dataSet in backgroundDataSets:
            self.bgDataSamples.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            #n_bgDataSamples += 1
            #if n_bgDataSamples % 2 == 0:
            #    #print('one: ', dataSet.fileGlob)
            #    self.bgDataSamples_one.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))
            #else:
            #    #print('two: ', dataSet.fileGlob)
            #    self.bgDataSamples_two.append(DataSample(dataSet, nEpoch, batchSize, variables, self.inputDataQueue, self.bgScaleSum, signal=False, background=True, domain=dataSet.domain, ptReweight=ptReweight))


    def sig_iterator(self):
        #print('\n\nsig_iterator:')
        for ds in self.sigDataSamples:
            #print(ds.fileList)
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_tt(self):
        #print('\n\nbg_iterator_tt')
        #_N = np.random.permutation(len(self.bgDataSamples_tt))
        #for n in _N:
        #    ds = self.bgDataSamples_tt[n]
        for ds in self.bgDataSamples_tt:
            #print(ds.fileList)
            for data in ds.get_data():
                yield data
        return

    def bg_iterator_cr(self):
        #print('\n\nbg_iterator_cr')
        #_N = np.random.permutation(len(self.bgDataSamples_cr))
        #for n in _N:
        #    ds = self.bgDataSamples_cr[n]
        for ds in self.bgDataSamples_cr:
            #print(ds.fileList)
            for data in ds.get_data():
                yield data
        return

    def bg_iterator_15to20(self):
        for ds in self.bgDataSamples_15to20:
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_20to30(self):
        for ds in self.bgDataSamples_20to30:
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_30to50(self):
        for ds in self.bgDataSamples_30to50:
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_50to80(self):
        for ds in self.bgDataSamples_50to80:
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_80to120(self):
        for ds in self.bgDataSamples_80to120:
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_120to170(self):
        for ds in self.bgDataSamples_120to170:
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_170to300(self):
        for ds in self.bgDataSamples_170to300:
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_300to470(self):
        for ds in self.bgDataSamples_300to470:
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_470to600(self):
        for ds in self.bgDataSamples_470to600:
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_600to800(self):
        for ds in self.bgDataSamples_600to800:
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_800to1000(self):
        for ds in self.bgDataSamples_800to1000:
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_1000to1500(self):
        for ds in self.bgDataSamples_1000to1500:
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_1500to2000(self):
        for ds in self.bgDataSamples_1500to2000:
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_2000to2500(self):
        for ds in self.bgDataSamples_2000to2500:
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_2500to3000(self):
        for ds in self.bgDataSamples_2500to3000:
            for data in ds.get_data():
                yield data
        return
    
    def bg_iterator_mc(self):
        #print('\n\nbg_iterator_mc')
        _N = np.random.permutation(len(self.bgDataSamples_mc))
        for n in _N:
            ds = self.bgDataSamples_mc[n]
            #print('mc')
            #print(ds.fileList)
            for data in ds.get_data():
                yield data
        return
            
    """def bg_iterator_one(self):
        for ds in self.bgDataSamples_one:
            for data in ds.get_data():
                yield data
        return

    def bg_iterator_two(self):
        for ds in self.bgDataSamples_two:
            for data in ds.get_data():
                yield data
        return"""

    def data_iterator(self):
        for ds in self.sigDataSamples:
            for data in ds.get_data():
                yield data
        for ds in self.bgDataSamples:
            for data in ds.get_data():
                yield data
        return

    def startFileQueues(self, coord):
        threads = []
        for ds in self.sigDataSamples:
            threads.append(ds.fileQueue.startQueueProcess(coord))

        for ds in self.bgDataSamples:
            threads.append(ds.fileQueue.startQueueProcess(coord))

        return threads

    def startDataQueues(self, sess, coord):
        enqueueOps = []
        threads = []
        for ds in self.sigDataSamples:
            threads += ds.start_threads(sess, coord)
            enqueueOps += ds.getEnqueueOp(2048, ds.dataSet.nEnqueueThreads)
            
        for ds in self.bgDataSamples:
            threads += ds.start_threads(sess, coord)
            enqueueOps += ds.getEnqueueOp(2048, ds.dataSet.nEnqueueThreads)

        #print('threads:', threads)
        #print('enqueueOps:', enqueueOps)
        return threads, enqueueOps

    def launchQueueThreads(self, sess):

        #print('in launch queue threads')
        # Create a coordinator, launch the queue runner threads.
        #stage 1 data cooridnator manages final random shuffle queue
        self.coordS1 = tf.train.Coordinator()
        #stage 2 data cooridnator manages FIFO queues, custom runner queues, and file name queues 
        self.coordS2 = tf.train.Coordinator()

        #print('after self.coordS1(2)')
        self.threadsS2 = self.startFileQueues(self.coordS2)

        #print('after self.threadsS2')
        #ensure that the filename queues are all populated before continuing 
        sleep(2)

        dataThreads, enqueueOps = self.startDataQueues(sess, self.coordS2)
        #print('after dataThreads, enqueueOps')
        self.threadsS2 += dataThreads

        #print('after self.threadsS2 +=')
        #create tf queue runner to manage the final random shuffle queue
        self.qr = tf.compat.v1.train.QueueRunner(self.inputDataQueue, enqueueOps, queue_closed_exception_types=(tf.errors.OutOfRangeError, tf.errors.CancelledError))

        #print('after self.qr')
        self.threadsS1 = self.qr.create_threads(sess, coord=self.coordS1, start=True)
        #print('after self.threadsS1')

    def continueTrainingLoop(self):
        try:
            return not self.coordS2.should_stop()
        except AttributeError:
            print("Run launchQueueThreads before starting the training loop")

    def continueFlushingQueue(self):
        try:
            return not self.coordS1.should_stop()
        except AttributeError:
            print("Run launchQueueThreads before starting the training loop")

    def requestStop(self, e = None):
        try:
            if e == None:
                self.coordS1.request_stop()
                self.coordS2.request_stop()
            else:
                self.coordS1.request_stop(e)
                self.coordS2.request_stop(e)
        except AttributeError:
            print("Run launchQueueThreads before starting the training loop")

    def join(self):
        try:
            self.coordS1.join(self.threadsS1)
            self.coordS2.join(self.threadsS2)
        except AttributeError:
            print("Run launchQueueThreads before starting the training loop")
        

