from copy import deepcopy

from .RandomHelper import AliasDict


def get_alias():
    channels_ = AliasDict({"4mu": None, "4e": None, "2e2mu": None})
    channels_.add_alias("4mu", ["four_muon", "fourMuon", "FourMuon", "four_mu", "4muon", "4Muon", "4_muon", "4_Muon"])
    channels_.add_alias("4e", ["four_electron", "fourElectron", "FourElectron", "four_e", "four_el", "4electron", "4Electron", "4el", "4_el", "4_e",
                               "4_electron", "4_Electron"])
    channels_.add_alias("2e2mu", ["2_e_2_mu", "TwoElectronTwoMuon", "2el2mu", "2_electron_2_muon", "two_electron_two_muon", "DiElectronDiMuon",
                                  "DiMuonDiElectron", "2_mu_2_el", "TwoMuonTwoElectron", "2mu2el", "2mu2e", "2_muon_2_electron",
                                  "two_muon_two_electron", "DiMuonDiElectron"])
    
    years_ = AliasDict({"2011": None, "2012": None})
    years_.add_alias("2011", ["11"])
    years_.add_alias("2012", ["12"])
    
    data_types_ = AliasDict({"run": None, "mc": None, "mc_bac": None, "mc_sig": None})
    data_types_.add_alias("run", ["Run", "ru", "RUN", "data", "measurement"])
    data_types_.add_alias("mc", ["MC", "MonteCarlo", "simulation", "mc signal"])
    data_types_.add_alias("mc_bac", ["background mc", "mc background", "mc_back", "mc_background", "background_mc",
                                     "back_mc", "background"])
    data_types_.add_alias("mc_sig", ["signal mc", "mc signal", "mc_signal", "signal_mc", "H_to", "Higgs", "higgs", "signal"])
    
    return {"channels": deepcopy(channels_), "years": deepcopy(years_), "data types": deepcopy(data_types_)}
