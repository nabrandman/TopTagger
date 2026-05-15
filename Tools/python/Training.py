import os
import errno
import optparse
from MVATrainers import mainTF
from taggerOptions import *

if __name__ == '__main__':

  #Option parsing 
  parser = getParser() 

  parser.add_option ('-C', "--checkpoint", dest='from_checkpoint', help='Checkpoint number from which to resume training', default='0')
  parser.add_option ('-L', "--label", dest='label', help='The name of the hdf5 dataset', default='')
  parser.add_option ('-S', "--saveName", dest='saveName', help='The name the model should be saved under', default='')
  parser.add_option ('-Q', "--halforquarter", dest='halforquarter', help='Whether the training data should be split with signal as half or a quarter of the total data', default='4')

  cmdLineOptions, args = parser.parse_args()

  if isinstance(cmdLineOptions.cfgFile, str): 
    options = taggerOptions.loadJSON(cmdLineOptions.cfgFile)
  else:
    options = taggerOptions.defaults()

  options = override(options,cmdLineOptions)

  #create output directory if it does not already exist 
  if len(options.runOp.directory):
    if options.runOp.directory[-1] != "/": options.runOp.directory += "/"
    try:
      os.mkdir(options.runOp.directory)
    except OSError as exc:
      if exc.errno == errno.EEXIST and os.path.isdir(options.runOp.directory):
        pass
      else:
        raise

  saveOptionsToJSON(options,options.runOp.directory+options.saveName)    
  mainTF(options, cmdLineOptions.label, cmdLineOptions.from_checkpoint, cmdLineOptions.saveName, cmdLineOptions.halforquarter)

  print("TRAINING DONE!")
