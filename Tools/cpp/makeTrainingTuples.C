#include "SusyAnaTools/Tools/samples.h"
#include "SusyAnaTools/Tools/NTupleReader.h"
#include "SusyAnaTools/Tools/MiniTupleMaker.h"
#include "SusyAnaTools/Tools/customize.h"
#include "SusyAnaTools/Tools/SATException.h"

#include "TopTagger/TopTagger/interface/TopTagger.h"
#include "TopTagger/TopTagger/interface/TopTaggerUtilities.h"
#include "TopTagger/TopTagger/interface/TopTaggerResults.h"
#include "TopTagger/CfgParser/interface/TTException.h"
#include "TaggerUtility.h"
#include "PlotUtility.h"

#include "TTree.h"
#include "TFile.h"
#include "Math/VectorUtil.h"

#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <set>
#include <math.h>
#include <memory>
#include <limits>

#include "hdf5.h"

class HDF5Writer
{
private:
    int nEvtsPerFile_, nEvts_, nFile_;
    std::string ofname_;
    std::map<std::string, std::vector<std::string>> variables_;
    std::map<std::string, std::vector<const char *>> variablesPtr_;
    std::map<std::string, std::vector<std::pair<bool,const  void*>>> pointers_;
    std::map<std::string, std::vector<float>> data_;

public:
    HDF5Writer(const std::map<std::string, std::vector<std::string>>& variables, int eventsPerFile, const std::string& ofname) : variables_(variables), nEvtsPerFile_(eventsPerFile), nEvts_(0), nFile_(0), ofname_(ofname)
    {
        for(const auto& varVec : variables_)
        {
            auto& ptrVec = variablesPtr_[varVec.first];
            for(const auto& str : varVec.second)
            {
                ptrVec.push_back(str.c_str());
            }
        }
    }

    void setTupleVars(const std::set<std::string>&) {}

    void initBranches(const NTupleReader& tr)
    {
      //std::cout << "in initBranches" << std::endl;
        for(const auto& dataset : variables_)
        {
	  //std::cout << "in for(const auto& dataset : variables_)" << std::endl;
            auto& ptrPair = pointers_[dataset.first];
	    //std::cout << "after auto& ptrPair = pointers_[dataset.first];" << std::endl;
            ptrPair.clear();
	    //std::cout << "after ptrPair.clear();" << std::endl;
            for(const auto& var : dataset.second)
            {
	      //std::cout << "in for(const auto& var : dataset.second)" << std::endl;
                std::string type;
                tr.getType(var, type);
		//std::cout << "after tr.getType(var, type);" << std::endl;
		//std::cout << "var: " << var << " | type: " << type << std::endl;
                if(type.find("vector") != std::string::npos)
                {
		  //std::cout << "in if(type.find(\"vector\") != std::string::npos)" << std::endl;
                    ptrPair.push_back(std::make_pair(true, tr.getVecPtr(var)));
		    //std::cout << "after ptrPair.push_back(std::make_pair(true, tr.getVecPtr(var)));" << std::endl;
                }
                else
                {
		  //std::cout << "in else" << std::endl;
                    ptrPair.push_back(std::make_pair(false, tr.getPtr(var)));
		    //std::cout << "after ptrPair.push_back(std::make_pair(false, tr.getPtr(var)));" << std::endl;
                }
            }
        }
    }

    void saveHDF5File()
    {
        herr_t      status;
        std::vector<char*> attr_data;

        std::string fileName(ofname_, 0, ofname_.find("."));
        fileName += "_" + std::to_string(nFile_) + ".h5";
        ++nFile_;

        /* Open an existing file. */
        hid_t file_id = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        for(const auto& data : data_)
        {
            const float* dset_data = data.second.data();
            /* Create the data space for the dataset. */
            hsize_t dims[2];
            dims[1] = variables_[data.first].size();
            dims[0] = data.second.size()/dims[1];
            hid_t dataspace_id = H5Screate_simple(2, dims, NULL);

            /* Create the dataset. */
            hid_t dataset_id = H5Dcreate2(file_id, data.first.c_str(), H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            /* Write the dataset. */
            status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);

            //create type
            hid_t vls_type_c_id = H5Tcopy(H5T_C_S1);
            status = H5Tset_size(vls_type_c_id, H5T_VARIABLE);

            /* Create a dataset attribute. */
            dims[0] = variables_[data.first].size();
            hid_t dataspace2_id = H5Screate_simple(1, dims, NULL);

            hid_t attribute_id = H5Acreate2 (dataset_id, "column_headers", vls_type_c_id, dataspace2_id, H5P_DEFAULT, H5P_DEFAULT);

            /* Write the attribute data. */
            status = H5Awrite(attribute_id, vls_type_c_id, variablesPtr_[data.first].data());

            /* Close the attribute. */
            status = H5Aclose(attribute_id);
            status = H5Sclose(dataspace2_id);

            /* End access to the dataset and release resources used by it. */
            status = H5Dclose(dataset_id);

            /* Terminate access to the data space. */ 
            status = H5Sclose(dataspace_id);
        }

        /* Close the file. */
        status = H5Fclose(file_id);
    }

    void fill()
    {
        ++nEvts_;
        size_t maxSize = 0;
        for(const auto& dataset : variables_)
        {
            auto& ptrPair = pointers_[dataset.first];
            auto& varVec = variables_[dataset.first];
            auto& dataVec = data_[dataset.first];
            int nCand = 0;
            //get ncand
            int iVar = 0;
            for(const auto& pp : ptrPair)
            {
                ++iVar;
                //Look for the first vector
                if(pp.first)
                {
                    nCand = (*static_cast<const std::vector<float> * const * const>(pp.second))->size();
                    break;
                }
            }
            for(int i = 0; i < nCand; ++i)
            {
                for(const auto& pp : ptrPair)
                {
                    //Check if this is a vector or a pointer 
                    if(pp.first) dataVec.push_back(nCand?((**static_cast<const std::vector<float> * const * const>(pp.second))[i]):0.0);
                    else         dataVec.push_back(*static_cast<const float * const>(pp.second));
                }
            }
            maxSize = std::max(maxSize, dataVec.size()/varVec.size());
        }

        if(maxSize >= nEvtsPerFile_)
        {
            //if we reached the max event per file lets write the file
            saveHDF5File();

            //reset the data structure 
            nEvts_ = 0;
            for(auto& data : data_) data.second.clear();
        }
    }

    ~HDF5Writer()
    {
        //Write the last events no matter what we have                                                                                                                                                                        
        saveHDF5File();
    }
    
};

class PrepVariables
{
private:
    template<typename T>
    class VariableHolder
    {
    private:
        NTupleReader* tr_;
    public:
        std::map<std::string, std::vector<T>*> variables_;

        void add(const std::string& key, const T& var)
        {
            if(variables_.find(key) == variables_.end() || variables_[key] == nullptr)
            {
                variables_[key] = new std::vector<T>();
            }

            variables_[key]->push_back(var);
        }

        void registerFunctions()
        {
            for(auto& entry : variables_)
            {
                if(entry.second == nullptr) entry.second = new std::vector<T>();
                tr_->registerDerivedVec(entry.first, entry.second);
            }
        }

        VariableHolder(NTupleReader& tr) : tr_(&tr) {}
    };
  
    TopTagger* topTagger_;
    TopCat topMatcher_;
    int eventNum_, bgPrescale_;
    const std::map<std::string, std::vector<std::string>>& variables_;
  //std::cout << "before mvaCalc_;" << std::endl;
  std::shared_ptr<ttUtility::MVAInputCalculator> mvaCalc_;

  //  std::shared_ptr<ttUtility::TrijetInputCalculator> mvaCalc_;
  //std::cout << "after mvaCalc_;"
    std::vector<float> values_;
    bool signal_;
    int Nbu_, Ncu_, Nlu_, Ngu_;
    int Nbl_, Ncl_, Nll_, Ngl_;

    bool prepVariables(NTupleReader& tr)
    {
      //std::cout << "in prepVariables(tr)" << std::endl;
        const std::vector<TLorentzVector>& jetsLVec  = tr.getVec_LVFromNano<float>("Jet");
        const std::vector<float>& btagUParTAK4B      = tr.getVec<float>("Jet_btagUParTAK4B");

        //New Tagger starts here
        //prep input object (constituent) vector
        ttUtility::ConstAK4Inputs<float> myConstAK4Inputs = ttUtility::ConstAK4Inputs<float>(jetsLVec, btagUParTAK4B);

        auto convertToDoubleandRegister = [](NTupleReader& tr, std::string name)
        {

	  std::string type;
	  std::vector<float>* doubleVec;
	  tr.getType(name, type);
	  //std::cout << "\nconvertToDoubleandRegister:" << std::endl;
	  //std::cout << "Name: " << name << "  |  Type: " << type << std::endl;
	  if (type.find("float") != std::string::npos) {
	    //std::cout << "\tin float:" << std::endl;
	    const std::vector<float>& inVec = tr.getVec<float>(name);
	    //for (int i = 0; i < 10; i++) {
	    //  std::cout << inVec[i] << " | ";
	    //};
	    //std::cout << std::endl;
	    doubleVec = new std::vector<float>(inVec.begin(), inVec.end());
	    //for (int i = 0; i < 10; i++) {
	    //  std::cout << doubleVec[0][i] << " | ";
	    //};
	    //std::cout << std::endl;
	  } else {
	    if (type.find("short") != std::string::npos) {
	      const std::vector<short>& inVec = tr.getVec<short>(name);
	      doubleVec = new std::vector<float>(inVec.begin(), inVec.end());
	      //std::cout << "\tin short:" << std::endl;
	      //for (int i = 0; i < 10; i++) {
	      //  std::cout << inVec[i] << " | ";
	      //};
	      //std::cout << std::endl;
	      //for (int i = 0; i < 10; i++) {
	      //  std::cout << doubleVec[0][i] << " | ";
	      //};
	      //std::cout << std::endl;

	    } else if (type.find("unsigned char") != std::string::npos) {
	      const std::vector<unsigned char>& inVec = tr.getVec<unsigned char>(name);
	      doubleVec = new std::vector<float>(inVec.begin(), inVec.end());
	      //std::cout << "\tin unsigned char:" << std::endl;
	      //for (int i = 0; i < 10; i++) {
	      //std::cout << inVec[i] << " | ";
	      //};
	      //std::cout << std::endl;
	      //for (int i = 0; i < 10; i++) {
	      //std::cout << doubleVec[0][i] << " | ";
	      //};
	      //std::cout << std::endl;
	    } else {
	      const std::vector<int>& inVec = tr.getVec<int>(name);
	      doubleVec = new std::vector<float>(inVec.begin(), inVec.end());
	      //std::cout << "\tin else:" << std::endl;
	      //for (int i = 0; i < 10; i++) {
	      //std::cout << inVec[i] << " | ";
	      //};
	      //std::cout << std::endl;
	      //for (int i = 0; i < 10; i++) {
	      //std::cout << doubleVec[0][i] << " | ";
	      //};
	      //std::cout << std::endl;

	    };
	    tr.registerDerivedVec(name+"ConvertedToDouble", doubleVec);
	  };
	  return doubleVec;
        };

        typedef std::pair<std::vector<TLorentzVector>, std::vector<std::vector<const TLorentzVector*>>> GenInfoType;
        std::unique_ptr<GenInfoType> genTopInfo(nullptr);
        if(tr.checkBranch("GenPart_statusFlags"))
        {
            const std::vector<TLorentzVector>& genDecayLVec = tr.getVec_LVFromNano<float>("GenPart");
            const std::vector<short>& genDecayStatFlag      = tr.getVec<short>("GenPart_statusFlags");
            const std::vector<int>& genDecayPdgIdVec        = tr.getVec<int>("GenPart_pdgId");
            const std::vector<short>& genDecayMomIdxVec     = tr.getVec<short>("GenPart_genPartIdxMother");

            genTopInfo.reset(new GenInfoType(std::move(ttUtility::GetTopdauGenLVecFromNano( genDecayLVec, genDecayPdgIdVec, genDecayStatFlag, genDecayMomIdxVec))));
            const std::vector<TLorentzVector>& hadGenTops = genTopInfo->first;
            const std::vector<std::vector<const TLorentzVector*>>& hadGenTopDaughters = genTopInfo->second;

            myConstAK4Inputs.addGenCollections( hadGenTops, hadGenTopDaughters );

            myConstAK4Inputs.addSupplamentalVector("partonFlavor",                          *convertToDoubleandRegister(tr, "Jet_partonFlavour"));
        }
        else
        {
            myConstAK4Inputs.addSupplamentalVector("partonFlavor",                          tr.createDerivedVec<float>("thisIsATempVec", jetsLVec.size()));
        }

        myConstAK4Inputs.addSupplamentalVector("recoJetschargedHadronEnergyFraction",   *convertToDoubleandRegister(tr, "Jet_chHEF"));
        myConstAK4Inputs.addSupplamentalVector("recoJetschargedEmEnergyFraction",       *convertToDoubleandRegister(tr, "Jet_chEmEF"));
        myConstAK4Inputs.addSupplamentalVector("recoJetsneutralEmEnergyFraction",       *convertToDoubleandRegister(tr, "Jet_neEmEF"));
        myConstAK4Inputs.addSupplamentalVector("recoJetsmuonEnergyFraction",            *convertToDoubleandRegister(tr, "Jet_muEF"));
        myConstAK4Inputs.addSupplamentalVector("recoJetsHFHadronEnergyFraction",        *convertToDoubleandRegister(tr, "Jet_hfHEF"));
        myConstAK4Inputs.addSupplamentalVector("recoJetsHFEMEnergyFraction",            *convertToDoubleandRegister(tr, "Jet_hfEmEF"));
        myConstAK4Inputs.addSupplamentalVector("recoJetsneutralEnergyFraction",         *convertToDoubleandRegister(tr, "Jet_neHEF"));
        //myConstAK4Inputs.addSupplamentalVector("ChargedMultiplicity",                   tr.getVec<float>("Jet_chMultiplicity"));
	myConstAK4Inputs.addSupplamentalVector("ChargedMultiplicity",                   *convertToDoubleandRegister(tr, "Jet_chMultiplicity"));
        myConstAK4Inputs.addSupplamentalVector("NeutralMultiplicity",                   *convertToDoubleandRegister(tr, "Jet_neMultiplicity"));
        myConstAK4Inputs.addSupplamentalVector("ElectronMultiplicity",                  *convertToDoubleandRegister(tr, "Jet_nElectrons"));
        myConstAK4Inputs.addSupplamentalVector("MuonMultiplicity",                      *convertToDoubleandRegister(tr, "Jet_nMuons"));
	myConstAK4Inputs.addSupplamentalVector("TotalMultiplicity",                     *convertToDoubleandRegister(tr, "Jet_nConstituents"));
	myConstAK4Inputs.addSupplamentalVector("btagUParTAK4CvB",                       *convertToDoubleandRegister(tr, "Jet_btagUParTAK4CvB"));
	myConstAK4Inputs.addSupplamentalVector("btagUParTAK4CvL",                       *convertToDoubleandRegister(tr, "Jet_btagUParTAK4CvL"));
	myConstAK4Inputs.addSupplamentalVector("btagUParTAK4CvNotB",                    *convertToDoubleandRegister(tr, "Jet_btagUParTAK4CvNotB"));
	myConstAK4Inputs.addSupplamentalVector("btagUParTAK4QvG",                       *convertToDoubleandRegister(tr, "Jet_btagUParTAK4QvG"));
	myConstAK4Inputs.addSupplamentalVector("btagUParTAK4SvCB",                      *convertToDoubleandRegister(tr, "Jet_btagUParTAK4SvCB"));
	myConstAK4Inputs.addSupplamentalVector("btagUParTAK4SvUDG",                     *convertToDoubleandRegister(tr, "Jet_btagUParTAK4SvUDG"));
	myConstAK4Inputs.addSupplamentalVector("btagUParTAK4UDG",                       *convertToDoubleandRegister(tr, "Jet_btagUParTAK4UDG"));
	myConstAK4Inputs.addSupplamentalVector("btagUParTAK4probb",                     *convertToDoubleandRegister(tr, "Jet_btagUParTAK4probb"));
	myConstAK4Inputs.addSupplamentalVector("btagUParTAK4probbb",                    *convertToDoubleandRegister(tr, "Jet_btagUParTAK4probbb"));


        std::vector<Constituent> constituents = ttUtility::packageConstituents(myConstAK4Inputs);

        //run tagger
        topTagger_->runTagger(constituents);

        //retrieve results
        const TopTaggerResults& ttr = topTagger_->getResults();
        const std::vector<TopObject>& topCands = ttr.getTopCandidates();

        //if there are no top candidates, discard this event and move to the next
        //if(topCands.size() == 0) return false;

        //Get gen matching results

        std::vector<float> *genMatchdR = new std::vector<float>();
        std::vector<float> *genMatchConst = new std::vector<float>();
        std::vector<float> *genMatchVec = new std::vector<float>();

        //Class which holds and registers vectors of variables
        VariableHolder<float> vh(tr);

        //prepare a vector of gen top pt
        int icandGen = 0;
        if(genTopInfo)
        {
            for(auto& genTop : genTopInfo->first) 
            {
                vh.add("candNumGen", icandGen++);
                vh.add("genTopPt", genTop.Pt());
            }
        }
        if(!genTopInfo || genTopInfo->first.size() == 0) 
        {
            tr.registerDerivedVec("genTopPt", new std::vector<float>());
            tr.registerDerivedVec("candNumGen", new std::vector<float>());
        }

        std::vector<float>* candNum = new std::vector<float>();
        int iTop = 0;
        //prepare reco top quantities
	//std::cout << "before for(const TopObject& topCand : topCands)" << std::endl;
	//int topcand_counter = 0;
	bool is_signal = false;
        for(const TopObject& topCand : topCands)
        {
	  //std::cout << "in for(const TopObject& topCand : topCands)" << std::endl;
	  //++topcand_counter;
	  //std::cout << "topcand_counter: " << topcand_counter << std::endl;
            const auto* bestMatch = topCand.getBestGenTopMatch(0.6, 3, 3);
            bool hasBestMatch = bestMatch !=  nullptr;
            float bestMatchPt = bestMatch?(bestMatch->Pt()):(-999.9);

            const auto& topConstituents = topCand.getConstituents();

            int NConstMatches = 0;
            for(const auto* constituent : topConstituents)
            {
	      //std::cout << "in for(const auto* constituent : topConstituents)" << std::endl;
	      //std::cout << "NConstMatches: " << NConstMatches << std::endl;
                auto iter = constituent->getGenMatches().find(bestMatch);
                if(iter != constituent->getGenMatches().end())
                {
                    ++NConstMatches;
                }
            }
	    //std::cout << "after for(const auto* constituent : topConstituents)" << std::endl;
	    //std::cout << "NConstMatches: " << NConstMatches << std::endl;

            //parton category
            int j1_parton = abs(static_cast<int>(topConstituents[0]->getExtraVar("partonFlavor")));
            int j2_parton = abs(static_cast<int>(topConstituents[1]->getExtraVar("partonFlavor")));
            int j3_parton = abs(static_cast<int>(topConstituents[2]->getExtraVar("partonFlavor")));

            int Nb = (j1_parton == 5) + (j2_parton == 5) + (j3_parton == 5);
            int Nc = (j1_parton == 4) + (j2_parton == 4) + (j3_parton == 4);
            int Ng = (j1_parton == 21) + (j2_parton == 21) + (j3_parton == 21);
            int Nl = (j1_parton < 4) + (j2_parton < 4) + (j3_parton < 4);

	    //std::cout << "signal_: " << signal_ << " | (hasBestMatch && NConstMatches == 3 ): " << (hasBestMatch && NConstMatches == 3);
	    //std::cout << " | !(hasBestMatch && NConstMatches == 3): " << !(hasBestMatch && NConstMatches == 3) << std::endl;
	    //std::cout << "Nbu_: " << Nbu_ << " | Nb: " << Nb << std::endl;
	    //std::cout << "Ncu_: " << Ncu_ << " | Nc: " << Nc << std::endl;
	    //std::cout << "Nlu_: " << Nlu_ << " | Nl: " << Nl << std::endl;
	    //std::cout << "Ngu_: " << Ngu_ << " | Ng: " << Ng << std::endl;
	    //std::cout << "Nbl_: " << Nbl_ << " | Nb: " << Nb << std::endl;
	    //std::cout << "Ncl_: " << Ncl_ << " | Nc: " << Nc << std::endl;
	    //std::cout << "Nll_: " << Nll_ << " | Nl: " << Nl << std::endl;
	    //std::cout << "Ngl_: " << Ngl_ << " | Ng: " << Ng << std::endl;
            //if((hasBestMatch && NConstMatches == 3) || bgPrescale_++ == 0)
            if( ((signal_)?( hasBestMatch && NConstMatches == 3 ):( !(hasBestMatch && NConstMatches == 3) ) )
                && (Nbu_ < 0 || Nb <= Nbu_)
                && (Ncu_ < 0 || Nc <= Ncu_)
                && (Nlu_ < 0 || Nl <= Nlu_)
                && (Ngu_ < 0 || Ng <= Ngu_)
                && (Nbl_ < 0 || Nb >= Nbl_)
                && (Ncl_ < 0 || Nc >= Ncl_)
                && (Nll_ < 0 || Nl >= Nll_)
                && (Ngl_ < 0 || Ng >= Ngl_)
                )
            {
	      is_signal = true;
	      //std::cout << "in big if statement" << std::endl;
	      //tr.registerDerivedVar("is_signal", true);
	      //passMVABaseline = true;
                const auto& varNames = variables_.find("reco_candidates")->second;
                candNum->push_back(static_cast<float>(iTop++));
                genMatchConst->push_back(NConstMatches);
                genMatchdR->push_back(hasBestMatch);
                genMatchVec->push_back(bestMatchPt);

                mvaCalc_->setPtr(values_.data());
		//std::cout << "after mvaCalc_->setPtr(vales_.data());" << std::endl;
                if(mvaCalc_->calculateVars(topCand, 0))
                {
		  //std::cout << "in if(mvaCalc_->calculateVars(topCand, 0))" << std::endl;
                    for(int i = 0; i < varNames.size(); ++i)
                    {
		      //std::cout << "var: " << varNames[i] << " | value: " << values_[i] << " | " << std::numeric_limits<std::remove_reference<decltype(values_.front())>::type>::max() << std::endl;
		      //std::cout << "in for(int i = 0; i < varNames.size(); ++i)" << std::endl;
                        if(values_[i] < std::numeric_limits<std::remove_reference<decltype(values_.front())>::type>::max()) vh.add(varNames[i], values_[i]);
			//std::cout << "after if(values_[i] ...)" << std::endl;
                    }
		    //std::cout << "after for(int i = 0; i < varNames.size(); ++i)" << std::endl;
                }
		//std::cout << "after if(mvaCalc_->calculateVars(topCand, 0))" << std::endl;
            } else {
	      is_signal = false || is_signal; 
	    };
	    //tr.registerDerivedVar("is_signal", false);
	    //tr.registerDerivedVar("passMVABaseline", false);
	    //return true;
	    //};
	    //std::cout << "after if ( signal_ ...)" << std::endl;
            if(bgPrescale_ >= 1) bgPrescale_ = 0;
	    //std::cout << "after if(bgPrescale_ >=1) bgPrescale_ = 0;" << std::endl;
        }
	//std::cout << "after for(const TopObject& topCand : topCands)" << std::endl;

	tr.registerDerivedVar("is_signal", is_signal);
	if (is_signal) {
	  vh.registerFunctions();
	  std::cout << "after vh.registerFunctions();" << std::endl;
	  
	  //register matching vectors
	  tr.registerDerivedVec("genTopMatchesVec",        genMatchdR);
	  tr.registerDerivedVec("genConstiuentMatchesVec", genMatchConst);
	  tr.registerDerivedVec("genConstMatchGenPtVec",   genMatchVec);
	  
	  tr.registerDerivedVar("nConstituents", static_cast<int>(constituents.size()));
	  
	  tr.registerDerivedVar("eventNum", static_cast<float>(eventNum_++));
	  tr.registerDerivedVec("candNum", candNum);
	  tr.registerDerivedVar("ncand", static_cast<float>(candNum->size()));
	  
	  //Generate basic MVA selection 
	  bool passMVABaseline = true;//met > 100 && cntNJetsPt30 >= 5 && cntCSVS >= 1 && cntCSVL >= 2;//true;//(topCands.size() >= 1) || genMatches.second.second->size() >= 1;
	  tr.registerDerivedVar("passMVABaseline", passMVABaseline);
	} else {
	  bool passMVABaseline = false;
	  tr.registerDerivedVar("passMVABaseline", passMVABaseline);
	};
	
        return true;
    }

public:
    PrepVariables(const std::map<std::string, std::vector<std::string>>& variables, bool signal, int Nbl, int Ncl, int Nll, int Ngl, int Nbu, int Ncu, int Nlu, int Ngu) : variables_(variables), values_(variables.find("reco_candidates")->second.size(), std::numeric_limits<std::remove_reference<decltype(values_.front())>::type>::max()), signal_(signal), Nbu_(Nbu), Ncu_(Ncu), Nlu_(Nlu), Ngu_(Ngu), Nbl_(Nbl), Ncl_(Ncl), Nll_(Nll), Ngl_(Ngl)
    {
        eventNum_ = 0;
        bgPrescale_ = 0;

        topTagger_ = new TopTagger();
        topTagger_->setCfgFile("TopTaggerClusterOnly.cfg");

        mvaCalc_.reset(new ttUtility::TrijetInputCalculator());
        mvaCalc_->mapVars(variables_.find("reco_candidates")->second);
    }

    bool operator()(NTupleReader& tr)
    {
      //std::cout << "in bool operator()(NTupleReader& tr)" << std::endl;
        return prepVariables(tr);
    }
};

int main(int argc, char* argv[])
{
    using namespace std;

    int opt;
    int option_index = 0;
    static struct option long_options[] = {
        {"fakerate",         no_argument, 0, 'f'},
	{"condor",           no_argument, 0, 'c'},
        {"dataSets",   required_argument, 0, 'D'},
        {"numFiles",   required_argument, 0, 'N'},
        {"startFile",  required_argument, 0, 'M'},
        {"numEvts",    required_argument, 0, 'E'},
        {"ratio"  ,    required_argument, 0, 'R'},
        {"suffix"  ,   required_argument, 0, 'U'},

        {"nbl"  ,    required_argument, 0, 'b'},
        {"ncl"  ,    required_argument, 0, 'x'},
        {"nll"  ,    required_argument, 0, 'l'},
        {"ngl"  ,    required_argument, 0, 'g'},
        {"nbu"  ,    required_argument, 0, 'B'},
        {"ncu"  ,    required_argument, 0, 'C'},
        {"nlu"  ,    required_argument, 0, 'L'},
        {"ngu"  ,    required_argument, 0, 'G'},
        {"bg"  ,     no_argument,       0, 'S'},
    };
    bool forFakeRate = false;
    bool runOnCondor = false;
    bool signal = true;
    string dataSets = "", sampleloc = AnaSamples::fileDir, outFile = "Data/trainingTuple", sampleRatios = "1:1", label = "test";
    int nFiles = -1, startFile = 0, nEvts = -1, printInterval = 10000, Nbl = -1, Ncl = -1, Nll = -1, Ngl = -1, Nbu = -1, Ncu = -1, Nlu = -1, Ngu = -1;

    while((opt = getopt_long(argc, argv, "fcD:N:M:E:R:B:C:L:G:Sb:x:l:g:U:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
        case 'f':
            forFakeRate = true;
            break;

        case 'c':
            runOnCondor = true;
            break;

        case 'D':
            dataSets = optarg;
            break;

        case 'N':
            nFiles = int(atoi(optarg));
            break;

        case 'M':
            startFile = int(atoi(optarg));
            break;

        case 'E':
            nEvts = int(atoi(optarg)) - 1;
            break;

        case 'R':
            sampleRatios = optarg;
            break;

        case 'B':
            Nbu = int(atoi(optarg));
            break;

        case 'C':
            Ncu = int(atoi(optarg));
            break;

        case 'L':
            Nlu = int(atoi(optarg));
            break;

        case 'G':
            Ngu = int(atoi(optarg));
            break;

        case 'b':
            Nbl = int(atoi(optarg));
            break;

        case 'x':
            Ncl = int(atoi(optarg));
            break;

        case 'l':
            Nll = int(atoi(optarg));
            break;

        case 'g':
            Ngl = int(atoi(optarg));
            break;

        case 'S':
            signal = false;
            break;

        case 'U':
            label = optarg;
            break;
        }
    }

    //if running on condor override all optional settings
    if(runOnCondor)
    {
        char thistFile[128];
        sprintf(thistFile, "trainingTuple_%d", startFile);
        outFile = thistFile;
        sampleloc = "condor";
    }

    AnaSamples::SampleSet        ss("sampleSets_PostProcessed_2016.cfg", runOnCondor);
    AnaSamples::SampleCollection sc("sampleCollections_2016.cfg", ss);

    map<string, vector<AnaSamples::FileSummary>> fileMap;

    //Select approperiate datasets here
    if(dataSets.compare("TEST") == 0)
    {
        fileMap["DYJetsToLL"]  = {ss["DYJetsToLL_HT_600toInf"]};
        fileMap["ZJetsToNuNu"] = {ss["ZJetsToNuNu_HT_2500toInf"]};
        fileMap["DYJetsToLL_HT_600toInf"] = {ss["DYJetsToLL_HT_600toInf"]};
        fileMap["ZJetsToNuNu_HT_2500toInf"] = {ss["ZJetsToNuNu_HT_2500toInf"]};
        fileMap["TTbarDiLep"] = {ss["TTbarDiLep"]};
        fileMap["TTbarNoHad"] = {ss["TTbarDiLep"]};
    }
    else
    {
        if(ss[dataSets] != ss.null())
        {
            fileMap[dataSets] = {ss[dataSets]};
            //for(const auto& colls : ss[dataSets].getCollections())
            //{
            //    fileMap[colls] = {ss[dataSets]};
            //}
        }
        else if(sc[dataSets] != sc.null())
        {
            fileMap[dataSets] = {sc[dataSets]};
        }
    }

    const std::map<std::string, std::vector<std::string>> variables =
    {
        {"gen_tops", {"eventNum", "candNumGen", "genTopPt", "sampleWgt"} },
        {"reco_candidates", {"eventNum",
                             "candNum",
                             "ncand",

			     "cand_p",
                             "cand_dThetaMin",
                             "cand_dThetaMax",
                             "cand_dRMax",
                             "cand_eta",
                             "cand_m",
                             "cand_phi",
                             "cand_pt",
                             //"cand_p",

                             "dTheta12",
                             "dTheta13",
                             "dTheta23",
                             "j12_m",
                             "j13_m",
                             "j23_m",

                             "j1_ChargedMultiplicity",
			     "j1_TotalMultiplicity",
			     "j1_btagUParTAK4B",
			     "j1_btagUParTAK4CvB",
			     "j1_btagUParTAK4CvL",
			     "j1_btagUParTAK4CvNotB",
			     "j1_btagUParTAK4QvG",
			     "j1_btagUParTAK4SvCB",
			     "j1_btagUParTAK4SvUDG",
			     "j1_btagUParTAK4UDG",
			     "j1_btagUParTAK4probb",
			     "j1_btagUParTAK4probbb",
                             "j1_ElectronMultiplicity",
                             "j1_MuonMultiplicity",
                             "j1_NeutralMultiplicity",
                             "j1_m",
                             "j1_p",
                             "j1_pt_lab",
                             "j1_recoJetsHFEMEnergyFraction",
                             "j1_recoJetsHFHadronEnergyFraction",
                             "j1_recoJetschargedEmEnergyFraction",
                             "j1_recoJetschargedHadronEnergyFraction",
                             "j1_recoJetsmuonEnergyFraction",
                             "j1_recoJetsneutralEmEnergyFraction",
                             "j1_recoJetsneutralEnergyFraction",
                             "j1_partonFlavor",

                             "j2_ChargedMultiplicity",
			     "j2_TotalMultiplicity",
			     "j2_btagUParTAK4B",
			     "j2_btagUParTAK4CvB",
			     "j2_btagUParTAK4CvL",
			     "j2_btagUParTAK4CvNotB",
			     "j2_btagUParTAK4QvG",
			     "j2_btagUParTAK4SvCB",
			     "j2_btagUParTAK4SvUDG",
			     "j2_btagUParTAK4UDG",
			     "j2_btagUParTAK4probb",
			     "j2_btagUParTAK4probbb",
                             "j2_ElectronMultiplicity",
                             "j2_MuonMultiplicity",
                             "j2_NeutralMultiplicity",
                             "j2_m",
                             "j2_p",
                             "j2_recoJetsHFEMEnergyFraction",
                             "j2_recoJetsHFHadronEnergyFraction",
                             "j2_recoJetschargedEmEnergyFraction",
                             "j2_recoJetschargedHadronEnergyFraction",
                             "j2_recoJetsmuonEnergyFraction",
                             "j2_recoJetsneutralEmEnergyFraction",
                             "j2_recoJetsneutralEnergyFraction",
                             "j2_partonFlavor",

                             "j3_ChargedMultiplicity",
			     "j3_TotalMultiplicity",
			     "j3_btagUParTAK4B",
			     "j3_btagUParTAK4CvB",
			     "j3_btagUParTAK4CvL",
			     "j3_btagUParTAK4CvNotB",
			     "j3_btagUParTAK4QvG",
			     "j3_btagUParTAK4SvCB",
			     "j3_btagUParTAK4SvUDG",
			     "j3_btagUParTAK4UDG",
			     "j3_btagUParTAK4probb",
			     "j3_btagUParTAK4probbb",
                             "j3_ElectronMultiplicity",
                             "j3_MuonMultiplicity",
                             "j3_NeutralMultiplicity",
                             "j3_m",
                             "j3_p",
                             "j3_recoJetsHFEMEnergyFraction",
                             "j3_recoJetsHFHadronEnergyFraction",
                             "j3_recoJetschargedEmEnergyFraction",
                             "j3_recoJetschargedHadronEnergyFraction",
                             "j3_recoJetsmuonEnergyFraction",
                             "j3_recoJetsneutralEmEnergyFraction",
                             "j3_recoJetsneutralEnergyFraction",
                             "j3_partonFlavor",

                             "genTopMatchesVec",
                             "genConstiuentMatchesVec",
                             "genConstMatchGenPtVec",
                             "sampleWgt",
                             } }
    };

    //parse sample splitting and set up minituples
    vector<pair<std::unique_ptr<HDF5Writer>, int>> mtmVec;
    int sumRatio = 0;
    std::string is_signal = signal ? ("_sig") : ("_bkg");
    for(size_t pos = 0, iter = 0; pos != string::npos; pos = sampleRatios.find(":", pos + 1), ++iter)
    {
        int splitNum = stoi(sampleRatios.substr((pos)?(pos + 1):(0)));
        sumRatio += splitNum;
        string ofname;
        if(iter == 0)      ofname = outFile + "_" + label + "_division_" + to_string(iter) + "_" + dataSets + is_signal + "_training" + ".root";
        else if(iter == 1) ofname = outFile + "_" + label + "_division_" + to_string(iter) + "_" + dataSets + is_signal + "_validation" + ".root";
        else if(iter == 2) ofname = outFile + "_" + label + "_division_" + to_string(iter) + "_" + dataSets + is_signal + "_test" + ".root";
        else               ofname = outFile + "_" + label + "_division_" + to_string(iter) + "_" + dataSets + is_signal + ".root";
        mtmVec.emplace_back(std::unique_ptr<HDF5Writer>(new HDF5Writer(variables, 500000, ofname)), splitNum);
    }

    for(auto& fileVec : fileMap)
    {
        for(auto& file : fileVec.second)
        {
            int startCount = 0, fileCount = 0, NEvtsTotal = 0;

            std::cout << fileVec.first << std::endl;
            file.readFileList();

            for(const std::string& fname : file.filelist_)
            {
                if(startCount++ < startFile) continue;
                if(nFiles > 0 && fileCount++ >= nFiles) break;

                if(nFiles > 0) NEvtsTotal = 0;
                else if(nEvts >= 0 && NEvtsTotal > nEvts) break;

                //open input file and tree
                TFile *f = TFile::Open(fname.c_str());

                if(!f)
                {
                    std::cout << "File \"" << fname << "\" not found!!!!!!" << std::endl;
                    continue;
                }
                TTree *t = (TTree*)f->Get(file.treePath.c_str());

                if(!t)
                {
                    std::cout << "Tree \"" << file.treePath << "\" not found in file \"" << fname << "\"!!!!!!" << std::endl;
                    continue;
                }
                std::cout << "\t" << fname << std::endl;
		
                try
                {
		  std::cout << "in try" << std::endl;
                    //Don't bother with activateBranches, take advantage of new on-the-fly branch allocation
                    NTupleReader tr(t);
		    std::cout << "after NTupleReader initialization" << std::endl;

                    //register variable prep class with NTupleReader
                    PrepVariables prepVars(variables, signal, Nbl, Ncl, Nll, Ngl, Nbu, Ncu, Nlu, Ngu);
		    std::cout << "after PrepVariables" << std::endl;
                    tr.registerFunction(prepVars);
		    std::cout << "after registerFunction" << std::endl;

                    int splitCounter = 0, mtmIndex = 0;

                    bool branchesInitialized = false;

		    std::cout << "before while" << std::endl;
                    while(tr.getNextEvent())
                    {
		      std::cout << "\n\nin while" << std::endl;
                        //Get sample lumi weight and correct for the actual number of events 
                        //This needs to happen before we ad sampleWgt to the mini tuple variables to save
                        float weight = file.getWeight();
			std::cout << "after float weight = file.getWeight();" << std::endl;
                        tr.registerDerivedVar("sampleWgt", weight);
			std::cout << "after tr.registerDerivedVar(\"sampleWgt\", weight);" << std::endl;

                        //If nEvts is set, stop after so many events
                        if(nEvts > 0 && NEvtsTotal > nEvts) break;
			std::cout << "after if(nEvts > 0 && NEvtsTotal > nEvts) break;" << std::endl;
                        if(tr.getEvtNum() % printInterval == 0) std::cout << "Event #: " << tr.getEvtNum() << std::endl;
			std::cout << "after if(tr.getEvtNum() % printInterval == 0)" << std::endl;

			const bool& pass_signal = tr.getVar<bool>("is_signal");
			bool other_pass_signal = pass_signal;
			std::cout << "!branchesInitialized: " << !branchesInitialized << " | pass_signal: " << pass_signal << " | other_pass_signal: " << other_pass_signal << std::endl;
                        //Things to run only on first event
                        if((!branchesInitialized) && (other_pass_signal))
                        {
			  std::cout << "\nin if(!branchesInitialized)" << std::endl;
                            try
                            {
			      std::cout << "in try" << std::endl;
                                //Initialize the mini tuple branches, needs to be done after first call of tr.getNextEvent()
                                for(auto& mtm : mtmVec)
                                {
				  std::cout << "in for(auto& mtm : mtmVec)" << std::endl;
                                    mtm.first->initBranches(tr);
				    std::cout << "after mtm.first->initBranches(tr);" << std::endl;
                                }
                                branchesInitialized = true;
				std::cout << "after branchesInitialized = True;" << std::endl;
                            }
                            catch(const SATException& e)
                            {
			      std::cout << "##############################################in catch(const SATException& e)" << std::endl;
                                //do nothing here - this is sort of hacky
                                continue;
                            }
                        }
			std::cout << "after try/catch" << std::endl;
                        //Get cut variable 
                        const bool& passMVABaseline = tr.getVar<bool>("passMVABaseline");

			//fill mini tuple
			bool passbaseline = passMVABaseline;
			// if(passMVABaseline)
			std::cout << "passbaseline: " << passbaseline << std::endl;
			if(passbaseline && other_pass_signal)
                        {
			  std::cout << "in if(passbaseline)" << std::endl;
                            mtmVec[mtmIndex].first->fill();
			    std::cout << "after mtmVec[mtmIndex].first->fill();" << std::endl;
                            ++splitCounter;
                            if(splitCounter == mtmVec[mtmIndex].second)
                            {
			      std::cout << "in if(splitCounter == mtmVec[mtmIndex].second)" << std::endl;
                                splitCounter = 0;
                                mtmIndex = (mtmIndex + 1)%mtmVec.size();
                            }
			    std::cout << "after if(splitCounter == mtmVec[mtmIndex].second)" << std::endl;
                            ++NEvtsTotal;
                        }
			std::cout << "after if(passbaseline)" << std::endl;

                    }
		    std::cout << "after while(tr->getNextevent())" << std::endl;

                    f->Close();
                }
                catch(const SATException& e)
                {
		  std::cout << "in catch(const SATException& e)" << std::endl;
                    cout << e << endl;
                    throw;
                }
                catch(const TTException& e)
                {
		  std::cout << "in catch(const TTException& e)" << std::endl;
                    cout << e << endl;
                    throw;
                }
                catch(const string& e)
                {
		  std::cout << "in catch(const string& e)" << std::endl;
                    cout << e << endl;
                    throw;
                }
            }
	    std::cout << "after for(fname)" << std::endl;
        }
	std::cout << "after for(file)" << std::endl;
    }
    std::cout << "after for(filevec)" << std::endl;

    for(auto& mtm : mtmVec) {
      std::cout << "in for(auto& mtm : mtmVec)" << std::endl;
      mtm.first.reset();
    };
    std::cout << "after for(auto& mtm : mtmVec)" << std::endl;
}
