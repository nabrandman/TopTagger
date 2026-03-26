#! /bin/sh

export PYTHONPATH=~/.local/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPATH=$CMSSW_BASE/src/TopTagger:$PYTHONPATH
CURRENT=$(pwd)
TOP_DIR=$CMSSW_BASE/src/TopTagger
SUSY_DIR=$CMSSW_BASE/src/SusyAnaTools
JSON_DIR=$CMSSW_BASE/src/json
TOP_GIT="git clone git@github.com:susy2015/TopTagger.git"
SUSY_GIT="git clone git@github.com:susy2015/SusyAnaTools.git"
JSON_GIT="git clone git@github.com:nlohmann/json.git"


if [ ! -d "$TOP_DIR" ]; then
    echo "I don't even know how you did this but the Top Tagger isn't here"
    echo "Please acquire the Top Tagger and place it in your CMSSW release src directory"
    echo "You can do this with:"
    echo ""
    echo $TOP_GIT
    echo ""
    exit 1
fi

if [ -d "$SUSY_DIR" ]; then
    echo "SusyAnaTools found"
else
    echo "SusyAnaTools NOT FOUND"
    echo "Please acquire the SusyAnaTools Package and place in your CMSSW release src directory"
    echo "This package can be acquired with the command:"
    echo ""
    echo $SUSY_GIT
    echo ""
    echo "afterwards don't forget to get the json package as well if you haven't yet"
    echo ""
    echo "That one can be acquired with:"
    echo ""
    echo $JSON_GIT
    echo ""
    echo "you may need to remove the \"test\" directory from this repo after you've cloned it for this to compile" 
    echo ""
    echo "after you've acquired all necessary dependencies then try compiling CMSSW with scram and then rerun this script"
    exit 2
fi

if [ -d "$JSON_DIR" ]; then
    echo "json found"
else
    echo "json NOT FOUND"
    echo "Please acquire the json Package and place in your CMSSW release src directory"
    echo "This package can be acquired with the command"
    echo ""
    echo $JSON_GIT
    echo "you may need to remove the \"test\" directory from this repo after you've cloned it for this to compile"
    echo ""
    echo "afterwards don't forget to get the SusyAnaTools package as well if you haven't yet"
    echo ""
    echo "That one can be acquired with:"
    echo ""
    echo $SUSY_GIT
    echo ""
    echo "after you've acquired all necessary dependencies then try compiling CMSSW with scram and then rerun this script"
    exit 3
fi

echo "cd $CMSSW_BASE/src/TopTagger/TopTagger/test"
cd $CMSSW_BASE/src/TopTagger/TopTagger/test
echo "./configure"
./configure
echo ""
echo "make clean"
make clean
echo ""
echo "make -j8"
make -j8
echo ""
echo "source taggerSetup.sh"
. taggerSetup.sh
echo ""
echo "cd $CMSSW_BASE/src/TopTagger/Tools"
cd $CMSSW_BASE/src/TopTagger/Tools
echo "./configure"
./configure
echo ""
echo "make clean"
make clean
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
echo ""
echo "make -j8"
make -j8
echo ""
echo "source taggerSetup.sh"
. taggerSetup.sh
echo ""
echo "cd $CURRENT"
cd $CURRENT
pwd


