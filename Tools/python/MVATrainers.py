import numpy
#import pandas as pd
from DataGetter import DataGetter
from math import sqrt, exp
from time import sleep
from glob import glob

def getValidData(dg, validDataFiles, options):
  validDataArray = []

  #nSamples = 0
  #for dsn in validDataFiles:
  #  nSamples += dsn[1]

  minValidDataSize = 999999999
  for dsn in validDataFiles:
    validDataArray.append(dg.importData(samplesToRun = tuple(dsn[0]), ptReweight=False))

    arrayLen = len(validDataArray[-1]["data"])
    dataMultiplier = dsn[1]
    validDataSize = arrayLen/dataMultiplier
    if validDataSize < minValidDataSize:
      minValidDataSize = validDataSize
  
  validData = {}

  for data in validDataArray:
    for key in data:
      if key in validData:
        validData[key] = numpy.vstack([validData[key], data[key][:int(minValidDataSize*dataMultiplier)]])
      else:
        validData[key] = data[key][:int(minValidDataSize*dataMultiplier)]

  perm = numpy.random.permutation(validData["data"].shape[0])

  for key in validData:
    validData[key] = validData[key][perm]

  return validData

def combineValidationData(validDataSig, validDataBg):
  minNumalidData = min(len(validDataSig["data"]), len(validDataBg["data"]))
  
  validData = {}
  for key in validDataSig:
    validData[key] = numpy.vstack([validDataBg[key][:minNumalidData], validDataSig[key][:minNumalidData]])

  return validData

def mainTF(options, label, from_checkpoint):

  import tensorflow as tf
  from CreateModel import CreateModel
  from DataManager import DataManager
  from DataSet import DataSet
  import os
  import sys

  print("PROCESSING VALIDATION DATA")
  #tf.compat.v1.disable_eager_execution()

  print('tf.config.list_physical_devices: ', tf.config.list_physical_devices())

  #tf.debugging.set_log_device_placement(True)
  dgSig = DataGetter.DefinedVariables(options.netOp.vNames, signal = True)
  #dgBg = DataGetter.DefinedVariables(options.netOp.vNames, background = True)

  data_path = os.getcwd()
  if label.endswith('/'):
    label = label.split('/')[0]
  data_path = data_path+'/Data/'+label+'/'
  if label == '':
    label += 'model'
    data_path = data_path[:-1]

  vds = glob(data_path+'trainingTuple_*_'+label+'_division_1_TTToLNu2Q_sig_validation_*.h5')
  validDataSig = []
  for dfile in vds:
    #validDataSig.append( ((dfile, ), 1) )
    validDataSig.append( ((dfile, ), 1000) )

  #vdbt = glob(data_path+'trainingTuple_*_'+label+'_division_1_TTToLNu2Q_bkg_validation_*.h5')
  #validDataBgTTbar = []
  #for dfile in vdbt:
  #  validDataBgTTbar.append( ((dfile, ), 1) )

  #vdbqmc = glob(data_path+'trainingTuple_*_'+label+'_division_1_QCD_*_bkg_validation_*.h5')
  #validDataBgQCDMC = []
  #for dfile in vdbqmc:
  #  validDataBgQCDMC.append( ((dfile, ), 1) )

  #vdbqd = glob(data_path+'trainingTuple_*_'+label+'_division_1_QCDCR_bkg_validation_*.h5')
  #validDataBgQCDData = []
  #for dfile in vdbqd:
  #  validDataBgQCDData.append( ((dfile, ), 1) )

  # Import data
  #print(options.runOp.validationSamples)

  #validDataSig =       getValidData(dgSig, validDataSig,       options)
  validDataTTbar = getValidData(dgSig, validDataSig, options)
  #print('Acquired Signal Validation Samples')
  #validDataBgTTbar =   getValidData(dgBg,  validDataBgTTbar,   options)
  #print('Acquired Background TTbar Validation Samples')
  #validDataBgQCDMC =   getValidData(dgBg,  validDataBgQCDMC,   options)
  #print('Acquired Background QCD Monte Carlo Validation Samples')
  #validDataBgQCDData = getValidData(dgBg,  validDataBgQCDData, options)
  #print('Acquired Background QCD Control Region Validation Samples')

  #validDataTTbar = combineValidationData(validDataSig, validDataBgTTbar)
  print('Combined TTbar Validation Samples')
  #validDataQCDMC = combineValidationData(validDataSig, validDataBgQCDMC)
  #validDataQCDMC = validDataBgQCDMC
  #validDataQCDData = combineValidationData(validDataSig, validDataBgQCDData)
  #validDataQCDData = validDataBgQCDData

  #get input/output sizes
  nFeatures = validDataTTbar["data"].shape[1]
  nLabels = validDataTTbar["labels"].shape[1]
  nWeights = validDataTTbar["weights"].shape[1]

  #Training parameters
  l2Reg = options.runOp.l2Reg
  MiniBatchSize = options.runOp.minibatchSize
  nEpoch = options.runOp.nepoch
  ReportInterval = options.runOp.reportInterval
  validationCount = min(options.runOp.nValidationEvents, validDataTTbar["data"].shape[0])
  if validationCount <= 0:
    validationCount = validDataTTbar['data'].shape[0]

  #scale data inputs to mean 0, stddev 1
  categories = numpy.array(options.netOp.vCategories)
  mins = numpy.zeros(categories.shape, dtype=numpy.float32)
  ptps = numpy.zeros(categories.shape, dtype=numpy.float32)
  for i in range(categories.max()):
    selectedCategory = categories == i
    mins[selectedCategory] = validDataTTbar["data"][:,selectedCategory].mean()
    ptps[selectedCategory] = validDataTTbar["data"][:,selectedCategory].std()
  ptps[ptps < 1e-10] = 1.0

  ##Create data manager, this class controls how data is fed to the network for training
  #                 DataSet(fileGlob, xsec, Nevts, kFactor, sig, prescale, rescale)
  signalDataSets = [DataSet(data_path+"trainingTuple_*_"+label+"_division_0_TTToLNu2Q_sig_training_*.h5", 336.185965584, 480447813, 1.0, True, 0, 1.0, 1.0, 8)]
  signalValidSets = [DataSet(data_path+"trainingTuple_*_"+label+"_division_1_TTToLNu2Q_sig_validation_*.h5", 336.185965584, 480447813, 1.0, True, 0, 1.0, 1.0, 8)]
  print('Acquired Signal Training Sets')
  #pt reweighting histograms 
  ttbarRatio = (numpy.array([0.7976347,  1.010679,  1.0329635,  1.0712056,  1.1147588,  1.0072196,  0.79854023, 0.7216115,  0.7717652,  0.851551,   0.8372917 ]), numpy.array([  0.,  50., 100., 150., 200., 250., 300., 350., 400., 450., 500., 1e10]))
  QCDDataRatio = (numpy.array([0.50125164, 0.70985824, 1.007087,   1.6701245,  2.5925348,  3.6850858, 4.924969,   6.2674766,  7.5736594,  8.406105,   7.7529635 ]), numpy.array([  0.,  50., 100., 150., 200., 250., 300., 350., 400., 450., 500., 1e10]))
  QCDMCRatio = (numpy.array([0.75231355, 1.0563549,  1.2571484,  1.3007764,  1.0678109,  0.83444154, 0.641499,   0.49130705, 0.36807108, 0.24333349, 0.06963781]), numpy.array([  0.,  50., 100., 150., 200., 250., 300., 350., 400., 450., 500., 1e10]))

  backgroundDataSets = [DataSet(data_path+"trainingTuple_*_"+label+"_division_0_TTToLNu2Q_bkg_training_*.h5", 336.185965584, 480447813, 1.0, False, 0, 1.0, 1.0, 8),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCDCR_bkg_training_*.h5", 1.0, 1, 1.0, False, 1, 1.0, 1.0, 8),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCD_15to20_bkg_training_*.h5", 885700000.0, 99984691, 1.0, False, 2, 1.0, 1.0, 1),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCD_20to30_bkg_training_*.h5", 415700000.0, 99972336, 1.0, False, 2, 1.0, 1.0, 1),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCD_30to50_bkg_training_*.h5", 112300000.0, 99983302, 1.0, False, 2, 1.0, 1.0, 1),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCD_50to80_bkg_training_*.h5", 16730000.0, 97814968, 1.0, False, 2, 1.0, 1.0, 1),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCD_80to120_bkg_training_*.h5", 2506000.0, 99244066, 1.0, False, 2, 1.0, 1.0, 1),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCD_120to170_bkg_training_*.h5", 439800.0, 99786394, 1.0, False, 2, 1.0, 1.0, 1),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCD_170to300_bkg_training_*.h5", 113300.0, 99860965, 1.0, False, 2, 1.0, 1.0, 1),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCD_300to470_bkg_training_*.h5", 7581.0, 79821382, 1.0, False, 2, 1.0, 1.0, 1),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCD_470to600_bkg_training_*.h5", 623.3, 77529080, 1.0, False, 2, 1.0, 1.0, 1),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCD_600to800_bkg_training_*.h5", 178.7, 79708520, 1.0, False, 2, 1.0, 1.0, 1),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCD_800to1000_bkg_training_*.h5", 30.62, 79505640, 1.0, False, 2, 1.0, 1.0, 1),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCD_1000to1500_bkg_training_*.h5", 9.306, 79974618, 1.0, False, 2, 1.0, 1.0, 1),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCD_1500to2000_bkg_training_*.h5", 0.5015, 19997308, 1.0, False, 2, 1.0, 1.0, 1),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCD_2000to2500_bkg_training_*.h5", 0.04264, 19311060, 1.0, False, 2, 1.0, 1.0, 1),
                        DataSet(data_path+"trainingTuple_*_"+label+"_division_0_QCD_2500to3000_bkg_training_*.h5", 0.004454, 19996725, 1.0, False, 2, 1.0, 1.0, 1)]
  
  backgroundValidSets = [DataSet(data_path+"trainingTuple_*_"+label+"_division_1_TTToLNu2Q_bkg_validation_*.h5", 336.185965584, 480447813, 1.0, False, 0, 1.0, 1.0, 8),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCDCR_bkg_validation_*.h5", 1.0, 1, 1.0, False, 1, 1.0, 1.0, 8),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCD_15to20_bkg_validation_*.h5", 885700000.0, 99984691, 1.0, False, 2, 1.0, 1.0, 1),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCD_20to30_bkg_validation_*.h5", 415700000.0, 99972336, 1.0, False, 2, 1.0, 1.0, 1),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCD_30to50_bkg_validation_*.h5", 112300000.0, 99983302, 1.0, False, 2, 1.0, 1.0, 1),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCD_50to80_bkg_validation_*.h5", 16730000.0, 97814968, 1.0, False, 2, 1.0, 1.0, 1),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCD_80to120_bkg_validation_*.h5", 2506000.0, 99244066, 1.0, False, 2, 1.0, 1.0, 1),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCD_120to170_bkg_validation_*.h5", 439800.0, 99786394, 1.0, False, 2, 1.0, 1.0, 1),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCD_170to300_bkg_validation_*.h5", 113300.0, 99860965, 1.0, False, 2, 1.0, 1.0, 1),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCD_300to470_bkg_validation_*.h5", 7581.0, 79821382, 1.0, False, 2, 1.0, 1.0, 1),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCD_470to600_bkg_validation_*.h5", 623.3, 77529080, 1.0, False, 2, 1.0, 1.0, 1),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCD_600to800_bkg_validation_*.h5", 178.7, 79708520, 1.0, False, 2, 1.0, 1.0, 1),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCD_800to1000_bkg_validation_*.h5", 30.62, 79505640, 1.0, False, 2, 1.0, 1.0, 1),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCD_1000to1500_bkg_validation_*.h5", 9.306, 79974618, 1.0, False, 2, 1.0, 1.0, 1),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCD_1500to2000_bkg_validation_*.h5", 0.5015, 19997308, 1.0, False, 2, 1.0, 1.0, 1),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCD_2000to2500_bkg_validation_*.h5", 0.04264, 19311060, 1.0, False, 2, 1.0, 1.0, 1),
                         DataSet(data_path+"trainingTuple_*_"+label+"_division_1_QCD_2500to3000_bkg_validation_*.h5", 0.004454, 19996725, 1.0, False, 2, 1.0, 1.0, 1)]

  print('Acquired Background Training Sets')

  dm = DataManager(options.netOp.vNames, nEpoch, nFeatures, nLabels, 2, nWeights, options.runOp.ptReweight, signalDataSets, backgroundDataSets)
  dm_valid = DataManager(options.netOp.vNames, nEpoch, nFeatures, nLabels, 2, nWeights, options.runOp.ptReweight, signalValidSets, backgroundValidSets)
  print('Build DataManager')

  # Build the graph
  denseNetwork = [nFeatures]+options.netOp.denseLayers+[nLabels]
  convLayers = options.netOp.convLayers
  rnnNodes = options.netOp.rnnNodes
  rnnLayers = options.netOp.rnnLayers
  grw = 2/(1+exp(-i/10000.0)) - 1
  training=True
  mlp = CreateModel(options, denseNetwork, convLayers, rnnNodes, rnnLayers, mins, 1.0/ptps, l2Reg, options.runOp.keepProb, training, grw)
  print('Created MLP object')

  """def validDataGenerator():
    all_Valid_sets = [validDataTTbar, validDataQCDMC, validDataQCDData]
    counts = [len(validDataTTbar["data"]), len(validDataQCDMC["data"]), len(validDataQCDData["data"])]
    currentCount = 0
    thirdCount = 0
    while currentCount <= validationCount:
      if thirdCount == 3:
        thirdCount = 0
        currentCount += 1
      thirdCount += 1
      if counts[thirdCount - 1] <= currentCount:
        continue
      else:
        yield {'x': all_Valid_sets[thirdCount-1]["data"][currentCount], 'p_': all_Valid_sets[thirdCount-1]["domain"][currentCount], 'wgt': all_Valid_sets[thirdCount-1]["weights"][currentCount], 'y_':  all_Valid_sets[thirdCount-1]["labels"][currentCount]}
    return"""


  sig_data = tf.data.Dataset.from_generator(
    dm.sig_iterator,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  """bkg_data1 = tf.data.Dataset.from_generator(
    dm.bg_iterator_one,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data2 = tf.data.Dataset.from_generator(
    dm.bg_iterator_two,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )"""

  #bkg_data1 = bkg_data.enumerate() \
  #                    .filter(lambda i, data: i % 2 == 0) \
  #                    .map(lambda j, datum: datum)
  #bkg_data2 = bkg_data.enumerate() \
  #                    .filter(lambda i, data: i % 2 != 0) \
  #                    .map(lambda j, datum: datum)
  #train_data = tf.data.Dataset.from_tensor_slices([sig_data, bkg_data1, bkg_data2])
  #train_data = train_data.interleave(lambda x: x, cycle_length=3, block_length=1, num_parallel_calls=tf.data.AUTOTUNE)

  bkg_data_tt = tf.data.Dataset.from_generator(
    dm.bg_iterator_tt,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_cr = tf.data.Dataset.from_generator(
    dm.bg_iterator_cr,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_15to20 = tf.data.Dataset.from_generator(
    dm.bg_iterator_15to20,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_20to30 = tf.data.Dataset.from_generator(
    dm.bg_iterator_20to30,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_30to50 = tf.data.Dataset.from_generator(
    dm.bg_iterator_30to50,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_50to80 = tf.data.Dataset.from_generator(
    dm.bg_iterator_50to80,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_80to120 = tf.data.Dataset.from_generator(
    dm.bg_iterator_80to120,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_120to170 = tf.data.Dataset.from_generator(
    dm.bg_iterator_120to170,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_170to300 = tf.data.Dataset.from_generator(
    dm.bg_iterator_170to300,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_300to470 = tf.data.Dataset.from_generator(
    dm.bg_iterator_300to470,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_470to600 = tf.data.Dataset.from_generator(
    dm.bg_iterator_470to600,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_600to800 = tf.data.Dataset.from_generator(
    dm.bg_iterator_600to800,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_800to1000 = tf.data.Dataset.from_generator(
    dm.bg_iterator_800to1000,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_1000to1500 = tf.data.Dataset.from_generator(
    dm.bg_iterator_1000to1500,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_1500to2000 = tf.data.Dataset.from_generator(
    dm.bg_iterator_1500to2000,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_2000to2500 = tf.data.Dataset.from_generator(
    dm.bg_iterator_2000to2500,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_2500to3000 = tf.data.Dataset.from_generator(
    dm.bg_iterator_2500to3000,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  bkg_data_mc = tf.data.Dataset.from_tensor_slices(
    [bkg_data_15to20,
     bkg_data_20to30,
     bkg_data_30to50,
     bkg_data_50to80,
     bkg_data_80to120,
     bkg_data_120to170,
     bkg_data_170to300,
     bkg_data_300to470,
     bkg_data_470to600,
     bkg_data_600to800,
     bkg_data_800to1000,
     bkg_data_1000to1500,
     bkg_data_1500to2000,
     bkg_data_2000to2500,
     bkg_data_2500to3000]
    )
  bkg_data_mc = bkg_data_mc.interleave(lambda x: x, cycle_length=15, block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
  train_data = tf.data.Dataset.from_tensor_slices([sig_data, bkg_data_tt, bkg_data_cr, bkg_data_mc])
  train_data = train_data.interleave(lambda x: x, cycle_length=4, block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
  train_data = train_data.shuffle(buffer_size=MiniBatchSize*32*4)
  train_data = train_data.batch(MiniBatchSize)
  train_data = train_data.prefetch(tf.data.AUTOTUNE)

  """valid_data = tf.data.Dataset.from_generator(
    validDataGenerator,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )"""
  valid_data = tf.data.Dataset.from_generator(
    dm_valid.data_iterator,
    output_signature=(
      {
        'x'  : tf.TensorSpec(shape=(64,), dtype=tf.float32),
        'p_' : tf.TensorSpec(shape=(2,), dtype=tf.float64),
        'wgt': tf.TensorSpec(shape=(1,), dtype=tf.float64),
        'y_' : tf.TensorSpec(shape=(2,), dtype=tf.float32),
      }
    )
  )

  
  valid_data = valid_data.batch(MiniBatchSize)
  valid_data = valid_data.prefetch(tf.data.AUTOTUNE)

  checkpoint_location = options.runOp.directory+'trained_models/checkpoints/'
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_location+label+'_{epoch:02d}-{val_loss:.2f}.keras',
  )

  class LossLogger_perBatch(tf.keras.callbacks.Callback):
    def __init__(self, n_batches):
      super().__init__()
      self.n_batches = n_batches

    def on_train_batch_end(self, batch, logs=None):
      if batch % self.n_batches == 0:
        loss = logs.get('loss')
        accuracy = logs.get('accuracy')
        print(f"\n\tBatch {batch}: loss = {loss:.6f} | accuracy = {accuracy:.6f}\n")

  logger_callback = LossLogger_perBatch(n_batches=ReportInterval)
  #logging_callback = tf.keras.callbacks.ProgbarLogger(count_mode='samples').on_train_batch_end
  if from_checkpoint == '0':
    print('Creating New Model')
    mlpModel = mlp.get_model()
  else:
    print('Loading Checkpoint '+from_checkpoint+' for model: '+label)
    custom_layers = mlp.get_layers()
    if len(from_checkpoint) < 2:
      epoch_num = '0' + from_checkpoint
    else:
      epoch_num = from_checkpoint
    ckpt_names = os.listdir(os.getcwd()+'/'+checkpoint_location)
    ckpt_name = label+'_'+epoch_num+'-'
    ckpt_val_loss = ''
    for name in ckpt_names:
      if ckpt_name in name:
        ckpt_val_loss = name.split('.keras')[0]
        break
    if ckpt_val_loss == '':
      sys.exit('Checkpoint '+from_checkpoint+' not found for model: '+label)
    
    ckpt_val_loss = ckpt_val_loss.split(ckpt_name)[1]
    
    checkpoint_name = os.getcwd()+'/'+checkpoint_location+ckpt_name+ckpt_val_loss+'.keras'
    mlpModel = tf.keras.models.load_model(checkpoint_name, custom_objects=custom_layers)

  print('\n\nfitting')
  print("Reporting validation loss every %i batches with %i events per batch for %i epochs\n\n"%(ReportInterval, MiniBatchSize, nEpoch))
  history = mlpModel.fit(
    x=train_data,
    epochs=nEpoch,
    initial_epoch=int(from_checkpoint),
    #steps_per_epoch=ReportInterval,
    #verbose=2,
    validation_data=valid_data,
    callbacks=[logger_callback, checkpoint_callback],
  )

  print('TRAINING COMPLETE')
  mlpModel.save(options.runOp.directory+'trained_models/'+label+'_model.keras')
  print('model saved at: '+options.runOp.directory+'trained_models/'+label+'_model.keras')

  print(history)
  print(history.history)
  with open(options.runOp.directory+'trained_models/'+label+'_history.log', 'w') as f:
    for key in history.history.keys():
      f.write("\n\n"+key+"\n")
      for value in history.history[key]:
        f.write(str(value)+"\n")
