#! /bin/sh

export PYTHONPATH=~/.local/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPATH=$CMSSW_BASE/src/TopTagger:$PYTHONPATH

HDF5_LIB="/cvmfs/cms.cern.ch/slc7_amd64_gcc900/external/hdf5/1.8.17/lib/"
if [[ ":$LD_LIBRARY_PATH:" != *":$HDF5_LIB:"* ]];then
    echo ""
    echo "hdf5 libs not in path"
    echo "export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:/cvmfs/cms.cern.ch/slc7_amd64_gcc900/external/hdf5/1.8.17/lib/\""
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/cvmfs/cms.cern.ch/slc7_amd64_gcc900/external/hdf5/1.8.17/lib/"
else
    echo ""
    echo "hdf5 libs already in path"
fi

. $CMSSW_BASE/src/TopTagger/TopTagger/test/taggerSetup.sh

. $CMSSW_BASE/src/TopTagger/Tools/taggerSetup.sh
