#include "TopTagger/TopTagger/interface/TopTaggerUtilities.h"

#include "TopTagger/TopTagger/interface/TopTaggerResults.h"

#include "TopTagger/TopTagger/interface/lester_mt2_bisect.h"

#include <map>
#include <utility>
#include <regex>
#include <cstdlib>

namespace ttUtility
{
    ConstGenInputs::ConstGenInputs() : hadGenTops_(nullptr), hadGenTopDaughters_(nullptr) {}

    ConstGenInputs::ConstGenInputs(const std::vector<TLorentzVector>& hadGenTops, const std::vector<std::vector<const TLorentzVector*>>& hadGenTopDaughters) : hadGenTops_(&hadGenTops), hadGenTopDaughters_(&hadGenTopDaughters) {}

    void ConstGenInputs::addGenCollections(const std::vector<TLorentzVector>& hadGenTops, const std::vector<std::vector<const TLorentzVector*>>& hadGenTopDaughters)
    {
        hadGenTops_ = &hadGenTops;
        hadGenTopDaughters_ = &hadGenTopDaughters;
    }

    std::vector<Constituent> packageConstituents(const std::vector<TLorentzVector>& jetsLVec, const std::vector<double>& btagFactors)
    {
        return packageConstituents(ConstAK4Inputs<double>(jetsLVec, btagFactors));
    }


    double coreMT2calc(const TLorentzVector & fatJet1LVec, const TLorentzVector & fatJet2LVec, const TLorentzVector& metLVec)
    {
        // The input parameters associated with the particle
        // (or collection of particles) associated with the
        // first "side" of the event: 
        const double massOfSystemA =  fatJet1LVec.M(); // GeV
        const double pxOfSystemA   =  fatJet1LVec.Px(); // GeV
        const double pyOfSystemA   =  fatJet1LVec.Py(); // GeV
  
        // The input parameters associated with the particle
        // (or collection of particles) associated with the
        // second "side" of the event:
        const double massOfSystemB =  fatJet2LVec.M(); // GeV
        const double pxOfSystemB   =  fatJet2LVec.Px(); // GeV
        const double pyOfSystemB   =  fatJet2LVec.Py(); // GeV
  
        // The missing transverse momentum:
        const double pxMiss        = metLVec.Px(); // GeV
        const double pyMiss        = metLVec.Py(); // GeV
  
        // The mass of the "inivisible" particle presumed to have
        // been produced at the end of the decay chain in each
        // "half" of the event:    
        const double invis_mass    = metLVec.M(); // GeV

        double desiredPrecisionOnMt2 = 0; // Must be >=0.  If 0 alg aims for machine precision.  if >0, MT2 computed to supplied absolute precision.

        //asymm_mt2_lester_bisect::disableCopyrightMessage();

        double mt2 =  asymm_mt2_lester_bisect::get_mT2(
            massOfSystemA, pxOfSystemA, pyOfSystemA,
            massOfSystemB, pxOfSystemB, pyOfSystemB,
            pxMiss, pyMiss,
            invis_mass, invis_mass,
            desiredPrecisionOnMt2);

        return mt2;

    }

    double calculateMT2(const TopTaggerResults& ttr, const TLorentzVector& metLVec)
    {
        TLorentzVector fatJet1LVec(0, 0, 0,0);
        TLorentzVector fatJet2LVec(0, 0, 0,0);
        //Use result for top var
        const std::vector<TopObject*> &Ntop = ttr.getTops();  

        if (Ntop.size() == 0)
        {
            return 0.0;
        }

        if (Ntop.size() == 1)
        {
            fatJet1LVec = Ntop.at(0)->P();
            fatJet2LVec = ttr.getRsys().P();
     
            return coreMT2calc(fatJet1LVec, fatJet2LVec, metLVec);
        }

        if (Ntop.size() >= 2)
        {
            std::vector<double> cachedMT2vec;
            for(unsigned int it=0; it<Ntop.size(); it++)
            {
                for(unsigned int jt=it+1; jt<Ntop.size(); jt++)
                {
                    cachedMT2vec.push_back(coreMT2calc(Ntop.at(it)->P(), Ntop.at(jt)->P(), metLVec));
                } 
            }
            std::sort(cachedMT2vec.begin(), cachedMT2vec.end());

            return cachedMT2vec.front();
        }

        return 0.0;
    }

    inline double relu(const double x, const double bias = 0.0)
    {
      //std::cout << "in relu" << std::endl;
        return (x > bias)?x:0.0;
    }

    BDTMonojetInputCalculator::BDTMonojetInputCalculator()
    {
        ak8_sdmass_ = ak8_tau21_ = ak8_tau32_ = ak8_ptDR_ = ak8_rel_ptdiff_ = ak8_csv1_mass_ = ak8_csv1_csv_ = ak8_csv1_ptD_ = ak8_csv1_axis1_ = ak8_csv1_mult_ = ak8_csv2_mass_ = ak8_csv2_ptD_ = ak8_csv2_axis1_ = ak8_csv2_mult_ = -1;
    }

    void BDTMonojetInputCalculator::mapVars(const std::vector<std::string>& vars)
    {
        len_ = vars.size();

        for(unsigned int i = 0; i < vars.size(); ++i)
        {
            if(     vars[i].compare("ak8_sdmass") == 0)      ak8_sdmass_ = i;
            else if(vars[i].compare("ak8_tau21") == 0)       ak8_tau21_ = i;
            else if(vars[i].compare("ak8_tau32") == 0)       ak8_tau32_ = i;
            else if(vars[i].compare("ak8_ptDR") == 0)        ak8_ptDR_ = i;
            else if(vars[i].compare("ak8_rel_ptdiff") == 0)  ak8_rel_ptdiff_ = i;
            else if(vars[i].compare("ak8_csv1_mass") == 0)   ak8_csv1_mass_ = i;
            else if(vars[i].compare("ak8_csv1_csv") == 0)    ak8_csv1_csv_ = i;
            else if(vars[i].compare("ak8_csv1_ptD") == 0)    ak8_csv1_ptD_ = i;
            else if(vars[i].compare("ak8_csv1_axis1") == 0)  ak8_csv1_axis1_ = i;
            else if(vars[i].compare("ak8_csv1_mult") == 0)   ak8_csv1_mult_ = i;
            else if(vars[i].compare("ak8_csv2_mass") == 0)   ak8_csv2_mass_ = i;
            else if(vars[i].compare("ak8_csv2_ptD") == 0)    ak8_csv2_ptD_ = i;
            else if(vars[i].compare("ak8_csv2_axis1") == 0)  ak8_csv2_axis1_ = i;
            else if(vars[i].compare("ak8_csv2_mult") == 0)   ak8_csv2_mult_ = i;
        }
    }
        
    bool BDTMonojetInputCalculator::calculateVars(const TopObject& topCand, int iCand)
    {
        if(checkCand(topCand))
        {
            const auto& constituent = *topCand.getConstituents()[0];
            if(ak8_sdmass_ >= 0)     *(basePtr_ + ak8_sdmass_ + len_*iCand) = constituent.getSoftDropMass();
            if(ak8_tau21_ >= 0)      *(basePtr_ + ak8_tau21_ + len_*iCand) =  constituent.getTau1() > 0 ? constituent.getTau2()/constituent.getTau1() : 1e9;
            if(ak8_tau32_ >= 0)      *(basePtr_ + ak8_tau32_ + len_*iCand) =  constituent.getTau2() > 0 ? constituent.getTau3()/constituent.getTau2() : 1e9;

            const auto* sj1 = &constituent.getSubjets()[0];
            const auto* sj2 = &constituent.getSubjets()[1];
            double fj_deltaR = ROOT::Math::VectorUtil::DeltaR(sj1->p(), sj2->p());
            if(ak8_ptDR_ >= 0)       *(basePtr_ + ak8_ptDR_ + len_*iCand) =       fj_deltaR*constituent.p().Pt();
            if(ak8_rel_ptdiff_ >= 0) *(basePtr_ + ak8_rel_ptdiff_ + len_*iCand) = fabs(sj1->p().Pt() - sj2->p().Pt()) / constituent.p().Pt();
            if(sj1->getBTagDisc() < sj2->getBTagDisc()) std::swap(sj1,sj2);
            if(ak8_csv1_mass_ >= 0)  *(basePtr_ + ak8_csv1_mass_ + len_*iCand) =  sj1->p().M();
            if(ak8_csv1_csv_ >= 0)   *(basePtr_ + ak8_csv1_csv_ + len_*iCand) =   (sj1->getBTagDisc() > 0 ? sj1->getBTagDisc() : 0.);
            if(ak8_csv1_ptD_ >= 0)   *(basePtr_ + ak8_csv1_ptD_ + len_*iCand) =   sj1->getExtraVar("ptD");
            if(ak8_csv1_axis1_ >= 0) *(basePtr_ + ak8_csv1_axis1_ + len_*iCand) = sj1->getExtraVar("axis1");
            if(ak8_csv1_mult_ >= 0)  *(basePtr_ + ak8_csv1_mult_ + len_*iCand) =  sj1->getExtraVar("mult");
            if(ak8_csv2_mass_ >= 0)  *(basePtr_ + ak8_csv2_mass_ + len_*iCand) =  sj2->p().M();
            if(ak8_csv2_ptD_ >= 0)   *(basePtr_ + ak8_csv2_ptD_ + len_*iCand) =   sj2->getExtraVar("ptD");
            if(ak8_csv2_axis1_ >= 0) *(basePtr_ + ak8_csv2_axis1_ + len_*iCand) = sj2->getExtraVar("axis1");
            if(ak8_csv2_mult_ >= 0)  *(basePtr_ + ak8_csv2_mult_ + len_*iCand) =  sj2->getExtraVar("mult");

            return true;
        }
        return false;
    }

    bool BDTMonojetInputCalculator::checkCand(const TopObject& topCand)
    {
        return topCand.getNConstituents() == 1
            && topCand.getType() == TopObject::MERGED_TOP
            && topCand.getConstituents()[0]->getType() == Constituent::AK8JET
            && topCand.getConstituents()[0]->getSubjets().size() == 2;
    }

    BDTDijetInputCalculator::BDTDijetInputCalculator()
    {
        var_fj_sdmass_ = var_fj_tau21_ = var_fj_ptDR_ = var_fj_rel_ptdiff_ = var_sj1_ptD_ = var_sj1_axis1_ = var_sj1_mult_ = var_sj2_ptD_ = var_sj2_axis1_ = var_sj2_mult_ = var_sjmax_csv_ = var_sd_n2_ = -1;
    }

    void BDTDijetInputCalculator::mapVars(const std::vector<std::string>& vars)
    {
        len_ = vars.size();
        
        for(unsigned int i = 0; i < vars.size(); ++i)
        {
            if(     vars[i].compare("var_fj_sdmass") == 0)      var_fj_sdmass_ = i;
            else if(vars[i].compare("var_fj_tau21") == 0)       var_fj_tau21_ = i;
            else if(vars[i].compare("var_fj_ptDR") == 0)        var_fj_ptDR_ = i;
            else if(vars[i].compare("var_fj_rel_ptdiff") == 0)  var_fj_rel_ptdiff_ = i;
            else if(vars[i].compare("var_sj1_ptD") == 0)        var_sj1_ptD_ = i;
            else if(vars[i].compare("var_sj1_axis1") == 0)      var_sj1_axis1_ = i;
            else if(vars[i].compare("var_sj1_mult") == 0)       var_sj1_mult_ = i;
            else if(vars[i].compare("var_sj2_ptD") == 0)        var_sj2_ptD_ = i;
            else if(vars[i].compare("var_sj2_axis1") == 0)      var_sj2_axis1_ = i;
            else if(vars[i].compare("var_sj2_mult") == 0)       var_sj2_mult_ = i;
            else if(vars[i].compare("var_sjmax_csv") == 0)      var_sjmax_csv_ = i;
            else if(vars[i].compare("var_sd_n2") == 0)          var_sd_n2_ = i;
        }
    }
        
    bool BDTDijetInputCalculator::calculateVars(const TopObject& topCand, int iCand)
    {
        if(checkCand(topCand))
        {
            const auto* fatjet = topCand.getConstituents()[0];
            if(fatjet->getType() != Constituent::AK8JET) fatjet = topCand.getConstituents()[1];
            if(var_fj_sdmass_ >= 0)     *(basePtr_ + var_fj_sdmass_ + len_*iCand)   = fatjet->getSoftDropMass();
            if(var_fj_tau21_ >= 0)      *(basePtr_ + var_fj_tau21_ + len_*iCand)    = fatjet->getTau1() > 0 ? fatjet->getTau2()/fatjet->getTau1() : 1e9;
            // filling subjet variables
            if(fatjet->getSubjets().size() < 2) return false;
            const auto *sj1 = &fatjet->getSubjets()[0];
            const auto *sj2 = &fatjet->getSubjets()[1];
            double fj_deltaR =  ROOT::Math::VectorUtil::DeltaR(sj1->p(), sj2->p());
            if(var_fj_ptDR_ >= 0)       *(basePtr_ + var_fj_ptDR_ + len_*iCand)       = fj_deltaR*fatjet->p().Pt();
            if(var_fj_rel_ptdiff_ >= 0) *(basePtr_ + var_fj_rel_ptdiff_ + len_*iCand) = std::abs(sj1->p().Pt()-sj2->p().Pt())/fatjet->p().Pt();
            if(var_sj1_ptD_ >= 0)       *(basePtr_ + var_sj1_ptD_ + len_*iCand)       = sj1->getExtraVar("ptD");
            if(var_sj1_axis1_ >= 0)     *(basePtr_ + var_sj1_axis1_ + len_*iCand)     = sj1->getExtraVar("axis1");
            if(var_sj1_mult_ >= 0)      *(basePtr_ + var_sj1_mult_ + len_*iCand)      = sj1->getExtraVar("mult");
            if(var_sj2_ptD_ >= 0)       *(basePtr_ + var_sj2_ptD_ + len_*iCand)       = sj2->getExtraVar("ptD");
            if(var_sj2_axis1_ >= 0)     *(basePtr_ + var_sj2_axis1_ + len_*iCand)     = sj2->getExtraVar("axis1");
            if(var_sj2_mult_ >= 0)      *(basePtr_ + var_sj2_mult_ + len_*iCand)      = sj2->getExtraVar("mult");
            if(var_sjmax_csv_ >= 0)     *(basePtr_ + var_sjmax_csv_ + len_*iCand)     = std::max(std::max(sj1->getBTagDisc(),sj2->getBTagDisc()),0.0);
            if(var_sd_n2_ >= 0)
            {
                double var_sd_0 = sj2->p().Pt()/(sj1->p().Pt()+sj2->p().Pt());
                *(basePtr_ + var_sd_n2_ + len_*iCand)       = var_sd_0/std::pow(fj_deltaR,-2);
            }

            return true;
        }

        return false;
    }

    bool BDTDijetInputCalculator::checkCand(const TopObject& topCand)
    {
        return topCand.getNConstituents() == 2
            && topCand.getType() == TopObject::SEMIMERGEDWB_TOP;
    }

    TrijetInputCalculator::TrijetInputCalculator()
    {
        cand_pt_ = -1;
        cand_p_ = -1;
        cand_eta_ = -1;
        cand_phi_ = -1;
        cand_m_ = -1;
        cand_dRMax_ = -1;
        cand_dThetaMin_ = -1;
        cand_dThetaMax_ = -1;
        dRPtTop_ = -1;
        dRPtW_ = -1;
        sd_n2_ = -1;

        for(unsigned int i = 0; i < NCONST; ++i)
        {
            j_m_lab_[i] = -1;
            j_CSV_lab_[i] = -1;
            j_CvsL_lab_[i] = -1;
            dR12_lab_[i] = -1;
            dR12_3_lab_[i] = -1;
            j12_m_lab_[i] = -1;
            j_p_[i] = -1;
            j_p_top_[i] = -1;
            j_theta_top_[i] = -1;
            j_phi_top_[i] = -1;
            j_phi_lab_[i] = -1;
            j_eta_lab_[i] = -1;
            j_pt_lab_[i] = -1;
            j_m_[i] = -1;
            j_CSV_[i] = -1;
            j_recoJetsJecScaleRawToFull_[i] = -1;
            j_recoJetschargedHadronEnergyFraction_[i] = -1;
            j_recoJetschargedEmEnergyFraction_[i] = -1;
            j_recoJetsneutralEmEnergyFraction_[i] = -1;
            j_recoJetsmuonEnergyFraction_[i] = -1;
            j_recoJetsHFHadronEnergyFraction_[i] = -1;
            j_recoJetsHFEMEnergyFraction_[i] = -1;
            j_recoJetsneutralEnergyFraction_[i] = -1;
            j_ChargedMultiplicity_[i] = -1;
            j_NeutralMultiplicity_[i] = -1;
            j_ElectronMultiplicity_[i] = -1;
            j_MuonMultiplicity_[i] = -1;
	    j_TotalMultiplicity_[i] = -1;
	    j_btagUParTAK4CvB_[i] = -1;
	    j_btagUParTAK4CvL_[i] = -1;
	    j_btagUParTAK4CvNotB_[i] = -1;
	    j_btagUParTAK4QvG_[i] = -1;
	    j_btagUParTAK4SvCB_[i] = -1;
	    j_btagUParTAK4SvUDG_[i] = -1;
	    j_btagUParTAK4UDG_[i] = -1;
	    j_btagUParTAK4probb_[i] = -1;
	    j_btagUParTAK4probbb_[i] = -1;
            j_CombinedSvtx_[i] = -1;
            j_JetProba_[i] = -1;
            j_JetBprob_[i] = -1;
            j_recoJetsBtag_[i] = -1;
            j_recoJetsCharge_[i] = -1;
            j_partonFlavor_[i] = -1;
            dTheta_[i] = -1;
            j12_m_[i] = -1;
        }
    }

    void TrijetInputCalculator::mapVars(const std::vector<std::string>& vars)
    {
      //std::cout << "in MapVars" << std::endl;
        len_ = vars.size();

        for(unsigned int j = 0; j < vars.size(); ++j)
        {
	  if(vars[j].compare("cand_dThetaMin") == 0) {
	    //std::cout << "cand_dThetaMin found in variables at: " << j << std::endl;
	  };
            if(vars[j].compare("cand_pt") == 0) cand_pt_ = j;
            if(vars[j].compare("cand_p") == 0) cand_p_ = j;
            if(vars[j].compare("cand_eta") == 0) cand_eta_ = j;
            if(vars[j].compare("cand_phi") == 0) cand_phi_ = j;
            if(vars[j].compare("cand_m") == 0) cand_m_ = j;
            if(vars[j].compare("cand_dRMax") == 0) cand_dRMax_ = j;
            if(vars[j].compare("cand_dThetaMin") == 0) cand_dThetaMin_ = j;
            if(vars[j].compare("cand_dThetaMax") == 0) cand_dThetaMax_ = j;
            if(vars[j].compare("dRPtTop") == 0) dRPtTop_ = j;
            if(vars[j].compare("dRPtW") == 0) dRPtW_ = j;
            if(vars[j].compare("sd_n2") == 0) sd_n2_ = j;

            for(unsigned int i = 0; i < NCONST; ++i)
            {
                int iMin = std::min(i, (i+1)%NCONST);
                int iMax = std::max(i, (i+1)%NCONST);
                int iNNext = (i+2)%NCONST;

                if(vars[j].compare("j" + std::to_string(i + 1) + "_m_lab") == 0)                                  j_m_lab_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_CSV_lab") == 0)                                j_CSV_lab_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_CvsL_lab") == 0)                               j_CvsL_lab_[i] = j;
                if(vars[j].compare("dR" + std::to_string(iMin + 1) + std::to_string(iMax + 1) + "_lab") == 0)     dR12_lab_[i] = j;
                if(vars[j].compare("dR" + std::to_string(iNNext + 1) + "_" + std::to_string(iMin + 1) + std::to_string(iMax + 1) + "_lab") == 0) dR12_3_lab_[i] = j;
                if(vars[j].compare("j"  + std::to_string(iMin + 1) + std::to_string(iMax + 1) + "_m_lab") == 0)   j12_m_lab_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_p") == 0)                                      j_p_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_p_top") == 0)                                  j_p_top_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_theta_top") == 0)                              j_theta_top_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_phi_top") == 0)                                j_phi_top_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_phi_lab") == 0)                                j_phi_lab_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_eta_lab") == 0)                                j_eta_lab_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_pt_lab") == 0)                                 j_pt_lab_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_m") == 0)                                      j_m_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_CSV") == 0)                                    j_CSV_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_recoJetsJecScaleRawToFull") == 0)              j_recoJetsJecScaleRawToFull_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_recoJetschargedHadronEnergyFraction") == 0)    j_recoJetschargedHadronEnergyFraction_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_recoJetschargedEmEnergyFraction") == 0)        j_recoJetschargedEmEnergyFraction_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_recoJetsneutralEmEnergyFraction") == 0)        j_recoJetsneutralEmEnergyFraction_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_recoJetsmuonEnergyFraction") == 0)             j_recoJetsmuonEnergyFraction_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_recoJetsHFHadronEnergyFraction") == 0)         j_recoJetsHFHadronEnergyFraction_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_recoJetsHFEMEnergyFraction") == 0)             j_recoJetsHFEMEnergyFraction_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_recoJetsneutralEnergyFraction") == 0)          j_recoJetsneutralEnergyFraction_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_ChargedMultiplicity") == 0)                    j_ChargedMultiplicity_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_NeutralMultiplicity") == 0)                    j_NeutralMultiplicity_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_ElectronMultiplicity") == 0)                   j_ElectronMultiplicity_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_MuonMultiplicity") == 0)                       j_MuonMultiplicity_[i] = j;
		if(vars[j].compare("j" + std::to_string(i + 1) + "_TotalMultiplicity") == 0)                      j_TotalMultiplicity_[i] = j;
		if(vars[j].compare("j" + std::to_string(i + 1) + "_btagUParTAK4CvB") == 0)                        j_btagUParTAK4CvB_[i] = j;
		if(vars[j].compare("j" + std::to_string(i + 1) + "_btagUParTAK4CvL") == 0)                        j_btagUParTAK4CvL_[i] = j;
		if(vars[j].compare("j" + std::to_string(i + 1) + "_btagUParTAK4CvNotB") == 0)                     j_btagUParTAK4CvNotB_[i] = j;
		if(vars[j].compare("j" + std::to_string(i + 1) + "_btagUParTAK4QvG") == 0)                        j_btagUParTAK4QvG_[i] = j;
		if(vars[j].compare("j" + std::to_string(i + 1) + "_btagUParTAK4SvCB") == 0)                       j_btagUParTAK4SvCB_[i] = j;
		if(vars[j].compare("j" + std::to_string(i + 1) + "_btagUParTAK4SvUDG") == 0)                      j_btagUParTAK4SvUDG_[i] = j;
		if(vars[j].compare("j" + std::to_string(i + 1) + "_btagUParTAK4UDG") == 0)                        j_btagUParTAK4UDG_[i] = j;
		if(vars[j].compare("j" + std::to_string(i + 1) + "_btagUParTAK4probb") == 0)                      j_btagUParTAK4probb_[i] = j;
		if(vars[j].compare("j" + std::to_string(i + 1) + "_btagUParTAK4probbb") == 0)                     j_btagUParTAK4probbb_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_CombinedSvtx") == 0)                           j_CombinedSvtx_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_JetProba") == 0)                               j_JetProba_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_JetBprob") == 0)                               j_JetBprob_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_recoJetsBtag") == 0)                           j_recoJetsBtag_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_recoJetsCharge") == 0)                         j_recoJetsCharge_[i] = j;
                if(vars[j].compare("j" + std::to_string(i + 1) + "_partonFlavor") == 0)                           j_partonFlavor_[i] = j;
                if(vars[j].compare("dTheta" + std::to_string(iMin + 1) + std::to_string(iMax + 1)) == 0)          dTheta_[i] = j;
                if(vars[j].compare("j"   + std::to_string(iMin + 1) + std::to_string(iMax + 1) + "_m") == 0)      j12_m_[i] = j;
            }
        }
    }
        
    bool TrijetInputCalculator::calculateVars(const TopObject& topCand, int iCand)
    {
      //std::cout << "in TrijetInputCalculator::calculateVars" << std::endl;
        if(checkCand(topCand))
        {
	  //std::cout << "in if(checkCand(topCand))" << std::endl;
            //std::map<std::string, double> varMap;

            //Get top candidate variables
            if(cand_pt_ >= 0)        *(basePtr_ + cand_pt_ + len_*iCand)        = topCand.p().Pt();
            if(cand_p_ >= 0)         *(basePtr_ + cand_p_ + len_*iCand)         = topCand.p().P();
            if(cand_eta_ >= 0)       *(basePtr_ + cand_eta_ + len_*iCand)       = topCand.p().Eta();
            if(cand_phi_ >= 0)       *(basePtr_ + cand_phi_ + len_*iCand)       = topCand.p().Phi();
            if(cand_m_ >= 0)         *(basePtr_ + cand_m_ + len_*iCand)         = topCand.p().M();
            if(cand_dRMax_ >= 0)     *(basePtr_ + cand_dRMax_ + len_*iCand)     = topCand.getDRmax();
            if(cand_dThetaMin_ >= 0) *(basePtr_ + cand_dThetaMin_ + len_*iCand) = topCand.getDThetaMin();
            if(cand_dThetaMax_ >= 0) *(basePtr_ + cand_dThetaMax_ + len_*iCand) = topCand.getDThetaMax();

	    //std::cout << "after get top candidate variables" << std::endl;
            //Get Constituents
            //Get a copy instead of the reference
            std::vector<Constituent const *> top_constituents = topCand.getConstituents();

	    //std::cout << "after std::vector<Constituent const *> top_constituents = topcand.getConstitutents();" << std::endl;
            //resort by CSV
            std::sort(top_constituents.begin(), top_constituents.end(), [](const Constituent * const c1, const Constituent * const c2){ return c1->getBTagDisc() > c2->getBTagDisc(); });
	    //std::cout << "after std::sort" << std::endl;
            //switch candidates 2 and 3 if they are not in Pt ordering 
            if(top_constituents[2]->p().Pt() > top_constituents[1]->p().Pt())
            {
	      //std::cout << "in if(top_constituents[2]->p().Pt() > top_constituents[1]->p().Pt())" << std::endl;
                std::swap(top_constituents[1], top_constituents[2]);
		//std::cout << "after std::swap" << std::endl;
            }

	    //std::cout << "\ntop_constituents.size(): " << top_constituents.size() << std::endl;
            //Get constituent variables before deboost
            for(unsigned int i = 0; i < top_constituents.size(); ++i)
            {
	      //std::cout << "i: " << i << " in for(unsigned int i = 0; i < top_constituents.size(); ++i)" << std::endl;
                if(j_m_lab_[i] >= 0)       *(basePtr_ + j_m_lab_[i] + len_*iCand)       = top_constituents[i]->p().M();
		//std::cout << "after if(j_m_lab_[i] >= 0" << std::endl;
                if(j_CSV_lab_[i] >= 0)     *(basePtr_ + j_CSV_lab_[i] + len_*iCand)     = top_constituents[i]->getBTagDisc();
		//std::cout << "after if(j_CSV_lab_[i] >= 0)" << std::endl;
                if(j_CvsL_lab_[i] >= 0)     *(basePtr_ + j_CvsL_lab_[i] + len_*iCand)     = relu(top_constituents[i]->getExtraVar("CvsL"));
		//std::cout << "after if(j_CvsL_lab_[i] >= 0)" << std::endl;

                //index of next jet (assumes < 4 jets)
                unsigned int iNext = (i + 1) % top_constituents.size();
                unsigned int iNNext = (i + 2) % top_constituents.size();
		//std::cout << "after iNNext" << std::endl;
                //unsigned int iMin = std::min(i, iNext);
                //unsigned int iMax = std::max(i, iNext);

                //Calculate the angle variables
                if(dR12_lab_[i] >= 0)   *(basePtr_ + dR12_lab_[i] + len_*iCand)   = ROOT::Math::VectorUtil::DeltaR(top_constituents[i]->p(), top_constituents[iNext]->p());
		//std::cout << "after if(dR12_lab_[i] >= 0)" << std::endl;
                if(dR12_3_lab_[i] >= 0) *(basePtr_ + dR12_3_lab_[i] + len_*iCand) = ROOT::Math::VectorUtil::DeltaR(top_constituents[iNNext]->p(), top_constituents[i]->p() + top_constituents[iNext]->p());
		//std::cout << "after if(dR12_3_lab_[i] >= 0)" << std::endl;

                //calculate pair masses
                auto jetPair = top_constituents[i]->p() + top_constituents[iNext]->p();
		//std::cout << "after auto jetPair=" << std::endl;
                if(j12_m_lab_[i] >= 0) *(basePtr_ + j12_m_lab_[i] + len_*iCand) = jetPair.M();
		//std::cout << "after if(j12_m_lab_[i] >= 0)" << std::endl << std::endl;
            }

	    //std::cout << "after for(unsigned int i = 0; i < top_constituents.size(); ++i)" << std::endl;
            if(dRPtTop_ >= 0) *(basePtr_ + dRPtTop_ + len_*iCand) = ROOT::Math::VectorUtil::DeltaR(top_constituents[0]->p(), top_constituents[1]->p() + top_constituents[2]->p()) * topCand.p().Pt();
	    //std::cout << "after if(dRPtTop_ >= 0)" << std::endl;
            if(dRPtW_ >= 0) *(basePtr_ + dRPtW_ + len_*iCand) = ROOT::Math::VectorUtil::DeltaR(top_constituents[1]->p(), top_constituents[2]->p()) * (top_constituents[1]->p() + top_constituents[2]->p()).Pt();
	    //std::cout << "after if(dRPtW_ >=0)" << std::endl;
            if(sd_n2_ >= 0) 
            {
	      //std::cout << "in if(sd_n2_ >= 0)" << std::endl;
                double var_sd_0 = top_constituents[2]->p().Pt()/(top_constituents[1]->p().Pt()+top_constituents[2]->p().Pt());
		//std::cout << "after double var_sd_0" << std::endl;
                double var_WdR = ROOT::Math::VectorUtil::DeltaR(top_constituents[1]->p(), top_constituents[2]->p());
		//std::cout << "after double var_WdR" << std::endl;
                *(basePtr_ + sd_n2_ + len_*iCand) = var_sd_0 / pow(var_WdR, -2);
		//std::cout << "end if(sd_n2_ >= 0)" << std::endl;
            }
	    //std::cout << "after if(sd_n2_ >= 0)" << std::endl;

            std::vector<Constituent> RF_constituents;
	    //std::cout << "after std::vector<Constitent> RF_constituents;" << std::endl;

            for(const auto& constitutent : top_constituents)
            {
	      //std::cout << "in for(const auto& constitutent : top_constituents)" << std::endl;
                TLorentzVector p4(constitutent->p());
		//std::cout << "after TLorentzVector p4(constitutent->p());" << std::endl;
                p4.Boost(-topCand.p().BoostVector());
		//std::cout << "after p4.Boost(-topCand.p().BoostVector());" << std::endl;
                RF_constituents.emplace_back(*constitutent);
		//std::cout << "after RF_constitents.emplace_back(*constitutent);" << std::endl;
                RF_constituents.back().setP(p4);
		//std::cout << "after RF_contituents.back().setP(p4);" << std::endl << std::endl;
            }
	    //std::cout << "after for(const auto& constitutent : top_constituents)" << std::endl;

            //re-sort constituents by p after deboosting
            std::sort(RF_constituents.begin(), RF_constituents.end(), [](const Constituent& c1, const Constituent& c2){ return c1.p().P() > c2.p().P(); });
	    //std::cout << "after re-sort constituents by p after deboosting" << std::endl;

	    //std::cout << "b4 for(unsigned int i = 0; i < RF_constituents.size(); ++i)" << std::endl;
            //Get constituent variables
            for(unsigned int i = 0; i < RF_constituents.size(); ++i)
            {
	      //std::cout << "in for(unsigned int i = 0; i < RF.constituents.size(); ++i)" << std::endl;
                if(j_p_[i] >= 0) *(basePtr_ + j_p_[i] + len_*iCand)     = RF_constituents[i].p().P();
		//std::cout << "after if(j_p_[i] >= 0)" << std::endl;

                //This is a bit silly
                TLorentzVector p4(RF_constituents[i].p());
		//std::cout << "after TLorentzVector p4(RF_constituents[i].p());" << std::endl;
                p4.Boost(topCand.p().BoostVector());
		//std::cout << "after p4.Boost(..)" << std::endl;
		//std::cout << "what follows is a bunch of if statements" << std::endl;
                if(j_p_top_[i] >= 0)     *(basePtr_ + j_p_top_[i] + len_*iCand)     = p4.P();
		//std::cout << "j_p_top" << std::endl;
                if(j_theta_top_[i] >= 0) *(basePtr_ + j_theta_top_[i] + len_*iCand) = topCand.p().Angle(p4.Vect());
		//std::cout << "j_theta_top" << std::endl;
                if(j_phi_top_[i] >= 0)   *(basePtr_ + j_phi_top_[i] + len_*iCand)   = ROOT::Math::VectorUtil::DeltaPhi(RF_constituents[i].p(), RF_constituents[0].p());
		//std::cout << "j_phi_top" << std::endl;

                if(j_phi_lab_[i] >= 0) *(basePtr_ + j_phi_lab_[i] + len_*iCand)   = p4.Phi();
		//std::cout << "j_phi_lab" << std::endl;
                if(j_eta_lab_[i] >= 0) *(basePtr_ + j_eta_lab_[i] + len_*iCand)   = p4.Eta();
		//std::cout << "j_eta_lab" << std::endl;
                if(j_pt_lab_[i] >= 0)  *(basePtr_ + j_pt_lab_[i] + len_*iCand)    = p4.Pt();
		//std::cout << "j_pt_lab" << std::endl;
            
                if(j_m_[i] >= 0)   *(basePtr_ + j_m_[i] + len_*iCand)     = RF_constituents[i].p().M();
		//std::cout << "j_m" << std::endl;
                if(j_CSV_[i] >= 0) *(basePtr_ + j_CSV_[i] + len_*iCand)   = RF_constituents[i].getBTagDisc();
		//std::cout << "j_CSV" << std::endl;

                if(j_recoJetsJecScaleRawToFull_[i] >= 0)           *(basePtr_ + j_recoJetsJecScaleRawToFull_[i] + len_*iCand)           = relu(RF_constituents[i].getExtraVar("recoJetsJecScaleRawToFull"));
		//std::cout << "j_recoJetsJecScaleRawToFull" << std::endl;
                if(j_recoJetschargedHadronEnergyFraction_[i] >= 0) *(basePtr_ + j_recoJetschargedHadronEnergyFraction_[i] + len_*iCand) = relu(RF_constituents[i].getExtraVar("recoJetschargedHadronEnergyFraction"));
		//std::cout << "j_recoJetschargedHadronEnergyFraction" << std::endl;
                if(j_recoJetschargedEmEnergyFraction_[i] >= 0)     *(basePtr_ + j_recoJetschargedEmEnergyFraction_[i] + len_*iCand)     = relu(RF_constituents[i].getExtraVar("recoJetschargedEmEnergyFraction"));
		//std::cout << "j_recoJetschargedEmEnergyFraction" << std::endl;
                if(j_recoJetsneutralEmEnergyFraction_[i] >= 0)     *(basePtr_ + j_recoJetsneutralEmEnergyFraction_[i] + len_*iCand)     = relu(RF_constituents[i].getExtraVar("recoJetsneutralEmEnergyFraction"));
		//std::cout << "j_recoJetsneutralEmEnergyFraction" << std::endl;
                if(j_recoJetsmuonEnergyFraction_[i] >= 0)          *(basePtr_ + j_recoJetsmuonEnergyFraction_[i] + len_*iCand)          = relu(RF_constituents[i].getExtraVar("recoJetsmuonEnergyFraction"));
		//std::cout << "j_recoJetsmuonEnergyFraction" << std::endl;
                if(j_recoJetsHFHadronEnergyFraction_[i] >= 0)      *(basePtr_ + j_recoJetsHFHadronEnergyFraction_[i] + len_*iCand)      = relu(RF_constituents[i].getExtraVar("recoJetsHFHadronEnergyFraction"));
		//std::cout << "j_recoJetsHFHadronEnergyFraction" << std::endl;
                if(j_recoJetsHFEMEnergyFraction_[i] >= 0)          *(basePtr_ + j_recoJetsHFEMEnergyFraction_[i] + len_*iCand)          = relu(RF_constituents[i].getExtraVar("recoJetsHFEMEnergyFraction"));
		//std::cout << "j_recoJetsHFEMenergyFraction" << std::endl;
                if(j_recoJetsneutralEnergyFraction_[i] >= 0)       *(basePtr_ + j_recoJetsneutralEnergyFraction_[i] + len_*iCand)       = relu(RF_constituents[i].getExtraVar("recoJetsneutralEnergyFraction"));
		//std::cout << "j_recoJetsneutralEnergyFraction" << std::endl;
                if(j_ChargedMultiplicity_[i] >= 0)                 *(basePtr_ + j_ChargedMultiplicity_[i] + len_*iCand)                 = relu(RF_constituents[i].getExtraVar("ChargedMultiplicity"));
		//std::cout << "j_ChargedMultiplicity" << std::endl;
                if(j_NeutralMultiplicity_[i] >= 0)                 *(basePtr_ + j_NeutralMultiplicity_[i] + len_*iCand)                 = relu(RF_constituents[i].getExtraVar("NeutralMultiplicity"));
		//std::cout << "j_NeutralMultiplicity" << std::endl;
                if(j_ElectronMultiplicity_[i] >= 0)                *(basePtr_ + j_ElectronMultiplicity_[i] + len_*iCand)                = relu(RF_constituents[i].getExtraVar("ElectronMultiplicity"));
		//std::cout << "j_ElectronMultiplicity" << std::endl;
                if(j_MuonMultiplicity_[i] >= 0)                    *(basePtr_ + j_MuonMultiplicity_[i] + len_*iCand)                    = relu(RF_constituents[i].getExtraVar("MuonMultiplicity"));
		//std::cout << "j_MuonMultiplicity" << std::endl;
		if(j_TotalMultiplicity_[i] >= 0)                   *(basePtr_ + j_TotalMultiplicity_[i] + len_*iCand)                   = relu(RF_constituents[i].getExtraVar("TotalMultiplicity"));
		//std::cout << "j_TotalMultiplicity" << std::endl;
		//std::cout << "j_TotalMultiplicity_[i]: " << j_TotalMultiplicity_[i] << std::endl;

		//std::cout << "j_btagUParTAK4CvB_[i]: " << j_btagUParTAK4CvB_[i] << std::endl;
		//std::cout << "basePtr_: " << basePtr_ << " | len_: " << len_ << " | iCand: " << iCand << std::endl;
		if(j_btagUParTAK4CvB_[i] >= 0)                     *(basePtr_ + j_btagUParTAK4CvB_[i] + len_*iCand)                     = relu(RF_constituents[i].getExtraVar("btagUParTAK4CvB"));

		//std::cout << "j_btagUParTAK4CvB" << std::endl;
		if(j_btagUParTAK4CvL_[i] >= 0)                     *(basePtr_ + j_btagUParTAK4CvL_[i] + len_*iCand)                     = relu(RF_constituents[i].getExtraVar("btagUParTAK4CvL"));
		//std::cout << "j_btagUParTAK4CvL" << std::endl;
		if(j_btagUParTAK4CvNotB_[i] >= 0)                  *(basePtr_ + j_btagUParTAK4CvNotB_[i] + len_*iCand)                  = relu(RF_constituents[i].getExtraVar("btagUParTAK4CvNotB"));
		//std::cout << "j_btagUParTAK4CvNotB" << std::endl;
		if(j_btagUParTAK4QvG_[i] >= 0)                     *(basePtr_ + j_btagUParTAK4QvG_[i] + len_*iCand)                     = relu(RF_constituents[i].getExtraVar("btagUParTAK4QvG"));
		//std::cout << "j_btagUParTAK4QvG" << std::endl;
		if(j_btagUParTAK4SvCB_[i] >= 0)                    *(basePtr_ + j_btagUParTAK4SvCB_[i] + len_*iCand)                    = relu(RF_constituents[i].getExtraVar("btagUParTAK4SvCB"));
		//std::cout << "j_btagUParTAK4SvCB" << std::endl;
		if(j_btagUParTAK4SvUDG_[i] >= 0)                   *(basePtr_ + j_btagUParTAK4SvUDG_[i] + len_*iCand)                   = relu(RF_constituents[i].getExtraVar("btagUParTAK4SvUDG"));
		//std::cout << "j_btagUParTAK4SvUDG" << std::endl;
		if(j_btagUParTAK4UDG_[i] >= 0)                     *(basePtr_ + j_btagUParTAK4UDG_[i] + len_*iCand)                     = relu(RF_constituents[i].getExtraVar("btagUParTAK4UDG"));
		//std::cout << "j_btagUParTAK4UDG" << std::endl;
		if(j_btagUParTAK4probb_[i] >= 0)                   *(basePtr_ + j_btagUParTAK4probb_[i] + len_*iCand)                   = relu(RF_constituents[i].getExtraVar("btagUParTAK4probb"));
		//std::cout << "j_btagUParTAK4probb" << std::endl;
		if(j_btagUParTAK4probbb_[i] >= 0)                  *(basePtr_ + j_btagUParTAK4probbb_[i] + len_*iCand)                  = relu(RF_constituents[i].getExtraVar("btagUParTAK4probbb"));
		//std::cout << "j_btagUParTAK4probbb" << std::endl;
                if(j_CombinedSvtx_[i] >= 0)                        *(basePtr_ + j_CombinedSvtx_[i] + len_*iCand)                        = relu(RF_constituents[i].getExtraVar("CombinedSvtx"));
		//std::cout << "j_CombinedSvtx" << std::endl;
                if(j_JetProba_[i] >= 0)                            *(basePtr_ + j_JetProba_[i] + len_*iCand)                            = relu(RF_constituents[i].getExtraVar("JetProba"));
		//std::cout << "j_JetProba" << std::endl;
                if(j_JetBprob_[i] >= 0)                            *(basePtr_ + j_JetBprob_[i] + len_*iCand)                            = relu(RF_constituents[i].getExtraVar("JetBprob"));
		//std::cout << "j_JetBprob" << std::endl;
                if(j_recoJetsBtag_[i] >= 0)                        *(basePtr_ + j_recoJetsBtag_[i] + len_*iCand)                        = relu(RF_constituents[i].getExtraVar("recoJetsBtag"));
		//std::cout << "j_recoJetsBtag" << std::endl;
                if(j_recoJetsCharge_[i] >= 0)                      *(basePtr_ + j_recoJetsCharge_[i] + len_*iCand)                      = relu(RF_constituents[i].getExtraVar("recoJetsCharge"), -2);
		//std::cout << "j_recoJetsCharge" << std::endl;
                if(j_partonFlavor_[i] >= 0)                        *(basePtr_ + j_partonFlavor_[i] + len_*iCand)                        = relu(RF_constituents[i].getExtraVar("partonFlavor"), -22);
		//std::cout << "j_partonFlavor" << std::endl;
		//std::cout << "after bunch of if statements" << std::endl;

                //index of next jet (assumes < 4 jets)
                unsigned int iNext = (i + 1) % RF_constituents.size();
                unsigned int iMin = std::min(i, iNext);
                unsigned int iMax = std::max(i, iNext);
		//std::cout << "after get iNext, iMin, iMax" << std::endl;

                //Calculate delta angle variables
                if(dTheta_[i] >= 0) *(basePtr_ + dTheta_[i] + len_*iCand) = RF_constituents[iMin].p().Angle(RF_constituents[iMax].p().Vect());
		//std::cout << "after if(dTheta_[i] >= 0)" << std::endl;

                //calculate pair masses
                auto jetPair = RF_constituents[i].p() + RF_constituents[iNext].p();
		//std::cout << "after auto jetPair" << std::endl;
                if(j12_m_[i] >= 0) *(basePtr_ + j12_m_[i] + len_*iCand) = jetPair.M();
		//std::cout << "after if(j12_m_[i] >= 0)" << std::endl;
            }
	    //std::cout << "return true" << std::endl;
            return true;
        }
	//std::cout << "return false" << std::endl;    
        return false;
    }

    bool TrijetInputCalculator::checkCand(const TopObject& topCand)
    {
        return topCand.getNConstituents() == 3
            && topCand.getType() == TopObject::RESOLVED_TOP;
    }

    std::vector<std::string> getMVAVars()
    {
        return std::vector<std::string>({"genTopPt",
                    "MET",
                    "cand_dRMax",
                    "cand_dThetaMin",
                    "cand_dThetaMax",
                    "cand_eta",
                    "cand_m",
                    "cand_phi",
                    "cand_pt",
                    "cand_p",
                    "dR12_lab",
                    "dR13_lab",
                    "dR1_23_lab",
                    "dR23_lab",
                    "dR2_13_lab",
                    "dR3_12_lab",
                    "dRPtTop",
                    "dRPtW",
                    "dTheta12",
                    "dTheta13",
                    "dTheta23",
                    "j12_m",
                    "j12_m_lab",
                    "j13_m",
                    "j13_m_lab",

                    "j1_recoJetsJecScaleRawToFull",
                    "j1_recoJetschargedHadronEnergyFraction",
                    "j1_recoJetschargedEmEnergyFraction",
                    "j1_recoJetsneutralEmEnergyFraction",
                    "j1_recoJetsmuonEnergyFraction",
                    "j1_recoJetsHFHadronEnergyFraction",
                    "j1_recoJetsHFEMEnergyFraction",
                    "j1_recoJetsneutralEnergyFraction",
                    "j1_ChargedMultiplicity",
                    "j1_NeutralMultiplicity",
                    "j1_ElectronMultiplicity",
                    "j1_MuonMultiplicity",
	      "j1_TotalMultiplicity",
	            "j1_btagUParTAK4CvB",
	      "j1_btagUParTAK4CvL",
	      "j1_btagUParTAK4CvNotB",
	      "j1_btagUParTAK4QvG",
	      "j1_btagUParTAK4SvCB",
	      "j1_btagUParTAK4SvUDG",
	      "j1_btagUParTAK4UDG",
	      "j1_btagUParTAK4probb",
	      "j1_btagUParTAK4probbb",
                    "j1_CombinedSvtx",
                    "j1_JetProba",
                    "j1_JetBprob",
                    "j1_recoJetsBtag",
                    "j1_recoJetsCharge",
                    "j1_CSV",
                    "j1_CSV_lab",
                    "j1_eta_lab",
                    "j1_m",
                    "j1_m_lab",
                    "j1_p",
                    "j1_phi_lab",
                    "j1_pt_lab",

                    "j2_recoJetsJecScaleRawToFull",
                    "j2_recoJetschargedHadronEnergyFraction",
                    "j2_recoJetschargedEmEnergyFraction",
                    "j2_recoJetsneutralEmEnergyFraction",
                    "j2_recoJetsmuonEnergyFraction",
                    "j2_recoJetsHFHadronEnergyFraction",
                    "j2_recoJetsHFEMEnergyFraction",
                    "j2_recoJetsneutralEnergyFraction",
                    "j2_ChargedMultiplicity",
                    "j2_NeutralMultiplicity",
                    "j2_ElectronMultiplicity",
                    "j2_MuonMultiplicity",
	      "j2_TotalMultiplicity",
	      "j2_btagUParTAK4CvB",
	      "j2_btagUParTAK4CvL",
	      "j2_btagUParTAK4CvNotB",
	      "j2_btagUParTAK4QvG",
	      "j2_btagUParTAK4SvCB",
	      "j2_btagUParTAK4SvUDG",
	      "j2_btagUParTAK4UDG",
	      "j2_btagUParTAK4probb",
	      "j2_btagUParTAK4probbb",
                    "j2_CombinedSvtx",
                    "j2_JetProba",
                    "j2_JetBprob",
                    "j2_recoJetsBtag",
                    "j2_recoJetsCharge",
                    "j2_CSV",
                    "j2_CSV_lab",
                    "j2_eta_lab",
                    "j2_m",
                    "j2_m_lab",
                    "j2_p",
                    "j2_phi_lab",
                    "j2_pt_lab",

                    "j3_recoJetsJecScaleRawToFull",
                    "j3_recoJetschargedHadronEnergyFraction",
                    "j3_recoJetschargedEmEnergyFraction",
                    "j3_recoJetsneutralEmEnergyFraction",
                    "j3_recoJetsmuonEnergyFraction",
                    "j3_recoJetsHFHadronEnergyFraction",
                    "j3_recoJetsHFEMEnergyFraction",
                    "j3_recoJetsneutralEnergyFraction",
                    "j3_ChargedMultiplicity",
                    "j3_NeutralMultiplicity",
                    "j3_ElectronMultiplicity",
                    "j3_MuonMultiplicity",
	      "j3_TotalMultiplicity",
	      "j3_btagUParTAK4CvB",
	      "j3_btagUParTAK4CvL",
	      "j3_btagUParTAK4CvNotB",
	      "j3_btagUParTAK4QvG",
	      "j3_btagUParTAK4SvCB",
	      "j3_btagUParTAK4SvUDG",
	      "j3_btagUParTAK4UDG",
	      "j3_btagUParTAK4probb",
	      "j3_btagUParTAK4probbb",
                    "j3_CombinedSvtx",
                    "j3_JetProba",
                    "j3_JetBprob",
                    "j3_recoJetsBtag",
                    "j3_recoJetsCharge",
                    "j3_CSV",
                    "j3_CSV_lab",
                    "j3_eta_lab",
                    "j3_m",
                    "j3_m_lab",
                    "j3_p",
                    "j3_phi_lab",
                    "j3_pt_lab",

                    "j23_m",
                    "j23_m_lab",
                    "sd_n2",
                    "j1_p_top", "j1_theta_top", "j1_phi_top", "j2_p_top", "j2_theta_top", "j2_phi_top", "j3_p_top", "j3_theta_top", "j3_phi_top"});

    }


    std::vector<TLorentzVector> GetHadTopLVec(const std::vector<TLorentzVector>& genDecayLVec, const std::vector<int>& genDecayPdgIdVec, const std::vector<int>& genDecayIdxVec, const std::vector<int>& genDecayMomIdxVec)
    {
        std::vector<TLorentzVector> tLVec;
        for(unsigned it=0; it<genDecayLVec.size(); it++)
        {
            int pdgId = genDecayPdgIdVec.at(it);
            if(abs(pdgId)==6)
            {
                for(unsigned ig=0; ig<genDecayLVec.size(); ig++)
                {
                    if( genDecayMomIdxVec.at(ig) == genDecayIdxVec.at(it) )
                    {
                        int pdgId = genDecayPdgIdVec.at(ig);
                        if(abs(pdgId)==24)
                        {
                            int flag = 0;
                            for(unsigned iq=0; iq<genDecayLVec.size(); iq++)
                            {
                                if( genDecayMomIdxVec.at(iq) == genDecayIdxVec.at(ig) )
                                {
                                    int pdgid = genDecayPdgIdVec.at(iq);
                                    if(abs(pdgid)== 11 || abs(pdgid)== 13 || abs(pdgid)== 15) flag++;
                                }
                            }
                            if(!flag) tLVec.push_back(genDecayLVec.at(it));
                        }
                    }
                }//dau. loop
            }//top cond
        }//genloop
        return tLVec;
    }

    std::vector<const TLorentzVector*> GetTopdauLVec(const TLorentzVector& top, const std::vector<TLorentzVector>& genDecayLVec, const std::vector<int>& genDecayPdgIdVec, const std::vector<int>& genDecayIdxVec, const std::vector<int>& genDecayMomIdxVec)
    {
        std::vector<const TLorentzVector*>topdauLVec;
        for(unsigned it=0; it<genDecayLVec.size(); it++)
        {
            if(genDecayLVec[it]==top){
                for(unsigned ig=0; ig<genDecayLVec.size(); ig++)
                {
                    if( genDecayMomIdxVec.at(ig) == genDecayIdxVec.at(it) )
                    {
                        int pdgId = genDecayPdgIdVec.at(ig);
                        if(abs(pdgId)==5) topdauLVec.push_back(&(genDecayLVec[ig]));
                        if(abs(pdgId)==24)
                        {
                            //topdauLVec.push_back(genDecayLVec[ig]);
                            for(unsigned iq=0; iq<genDecayLVec.size(); iq++)
                            {
                                if( genDecayMomIdxVec.at(iq) == genDecayIdxVec.at(ig) )
                                {
                                    int pdgid = genDecayPdgIdVec.at(iq);
                                    if(abs(pdgid)!= 11 && abs(pdgid)!= 13 && abs(pdgid)!= 15) topdauLVec.push_back(&(genDecayLVec[iq]));
                                }
                            }
                        }
                    }
                }//dau. loop
            }//top cand.
        }//gen loop
        return topdauLVec;
    }

    void autoExpandEnvironmentVariables(std::string& path)
    {
        static std::regex env("\\$\\{([^}]+)\\}");
        std::smatch match;
        while (std::regex_search(path, match, env))
        {
            const char* s = getenv(match[1].str().c_str());
            const std::string var(s == NULL ? "" : s);
            path.replace(path.find(match[0]), match[0].length(), var);
        }
    }

}
