# Importando as Bibliotecas

import pprint
import json
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import anatools.data as data
import anatools.analysis as ana

from utils import signal_label, stack_sorting, position, process_signals

ana.start()
plt.style.use("default")



# Configuração

period = '18'
year_style = 2018
dataset_year = "2018"
basedir = '/home/thiagosousa/IC_Helena/Amostras/hhdmAnalysis_deepJet_Base/datasets'
region_id = "base"

# Setup output folders
dataset_name = basedir.split('/')[-2]
plots_path = f"./figures/{dataset_name}/{dataset_year}"
available_region_ids = ["base"]
for available_region_id in available_region_ids:
    Path(f"{plots_path}/regionID={available_region_id}").mkdir(parents=True, exist_ok=True)



# Ler Metadata

with open('/home/thiagosousa/IC_Helena/IC_Dark_Matter/hhdmAnalysis-IC/metadata.json', 'r') as f:
    metadata = json.load(f)

ST = metadata.get("datasets").get("ST")
TT = metadata.get("datasets").get("TT")
ZZ = metadata.get("datasets").get("ZZ")
WZ = metadata.get("datasets").get("WZ")
DY = metadata.get("datasets").get("DY")
RESIDUAL = metadata.get("datasets").get("RESIDUAL")
DATA = metadata.get("datasets").get("DATA")


# Ler os Datasets

ds = data.read_files(basedir, period, mode="normal")

data.join_datasets(ds, "ST", ST.get(period), mode="normal")
data.join_datasets(ds, "TT", TT.get(period), mode="normal")
data.join_datasets(ds, "ZZ", ZZ.get(period), mode="normal")
data.join_datasets(ds, "WZ", WZ.get(period), mode="normal")
data.join_datasets(ds, "DYJetsToLL", DY.get(period), mode="normal")
data.join_datasets(ds, "Residual", RESIDUAL.get(period), mode="normal")
data.join_datasets(ds, "Data", DATA.get(period), mode="normal")

print("Signal_400_100", ds["Signal_400_100"].shape)
print("Signal_1000_100", ds["Signal_1000_100"].shape)
print("ST", ds["ST"].shape)
print("TT", ds["TT"].shape)
print("ZZ", ds["ZZ"].shape)
print("WZ", ds["WZ"].shape)
print("DYJetsToLL", ds["DYJetsToLL"].shape)
print("Residual", ds["Residual"].shape)
print("Data", ds["Data"].shape)


# Preparando os Datasets para o Plot

colors_list = ["gainsboro", "orchid", "limegreen", "red", "skyblue", "darkgoldenrod"]
labels_list = [r"Residual SM", r"$WZ$", r"$ZZ$", "Single top", r"$t\bar{t}$", "Drell-Yan"]
bkg_list = ["Residual", "WZ", "ZZ", "ST", "TT", "DYJetsToLL"]
dataframes, labels, colors = stack_sorting(ds, colors_list, labels_list, bkg_list)


# Definindo as variáveis do Plot




# Signals to use in plot
signal_to_use = process_signals([
    ("Signal_400_100", "darkviolet"),
    ("Signal_1000_100", "blue")
])

# Variables to plot
variables = [
    ('Dijet_H_deltaPhi', r"Dijet $\Delta \phi$", 0, 6, 51, 5001),
    ('Dijet_H_pt', r"Dijet $H$ $p_T$ [GeV]", 0, 1000, 51, 5001),
    ('Dijet_M', r"Dijet $M$ [GeV]", 0, 1000, 51, 5001),
    ('Dijet_deltaEta', r"Dijet $\Delta \eta$", 0, 6, 51, 5001),
    ('Dijet_pt', r"Dijet $p_T$ [GeV]", 0, 1000, 51, 5001),
    ('HT', '$H_{T}$ [GeV]', 0, 1000, 51, 5001),
    ('Jet_abseta_max', r"Most Forward Jet $|\eta|$", 0, 5, 51, 5001),
    ('LeadingJet_eta', r"leading jet $|\eta|$", -3, 3, 51, 5001),
    ('LeadingJet_phi', r"leading jet $\phi$", -4, 4, 51, 5001),
    ('LeadingJet_pt', r"leading jet $p_{T}$ [GeV]", 0, 1000, 51, 5001),
    ('LeadingLep_eta', r"leading lepton $|\eta|$", 0, 3, 51, 5001),
    ('LeadingLep_pt', r"leading lepton $p_{T}$ [GeV]", 0, 1000, 51, 5001),
    ('LepLep_deltaM', r"$|M_{\ell \ell} - M_{Z}|$ [GeV]", 0, 30, 21, 5001),
    ('LepLep_deltaR', r"$\Delta R^{\ell \ell}$", 0, 4, 51, 5001),
    ('LepLep_eta', r"$|\eta|^{\ell \ell}$", 0, 5, 51, 5001),
    ('LepLep_mass', r"$M^{\ell \ell}$ [GeV]", 50, 130, 51, 5001),
    ("LepLep_phi", r"$\phi^{\ell \ell}$", -4, 4, 51, 5001),
    ('LepLep_pt', r"$p_{T}^{\ell \ell}$ [GeV]", 0, 1000, 51, 5001),
    ('MET_LepLep_Mt', r"$M^{\ell \ell, MET}_{T}$ [GeV]", 0, 1000, 51, 5001),
    ("MET_LepLep_deltaPhi", r"$\Delta \phi^{\ell \ell+MET}$", 0, 6, 51, 5001),
    ('MET_LepLep_deltaPt', r"$\Delta p^{\ell \ell, MET}_{T}$ [GeV]", 0, 30, 51, 5001),
    ("MET_phi", "MET $\phi$", -4, 4, 51, 5001),
    ('MET_pt',  r"MET $p_{T}$ [GeV]", 0, 1000, 51, 5001),
    ('MHT', '$M$ $H_{T}$ [GeV]', 0, 1000, 51, 5001),
    ("MT2LL", "MT2LL", 0, 700, 51, 5001),
    ("Nbjets", "Number of b-jets", 0, 10, 11, 5001),
    ("Njets", "Number of jets", 0, 10, 11, 5001),
    ('Njets_forward', 'Number of forward jets', 0, 10, 11, 5001),
    ('Njets_tight', 'Number of tight jets', 0, 10, 11, 5001),
    ('SubLeadingJet_eta', r"subleading jet $|\eta|$", -3, 3, 51, 5001),
    ('SubLeadingJet_phi', r"subleading jet $\phi$", -4, 4, 51, 5001),
    ('SubLeadingJet_pt', r"subleading jet $p_{T}$ [GeV]", 0, 1000, 51, 5001),
    ('TrailingLep_eta',  r"trailing lepton $|\eta|$", -3, 3, 51, 5001),
    ("TrailingLep_pt", r"trailing lepton $p_{T}$ [GeV]", 0, 700, 51, 5001)
]

# Plots das Distribuições

for var, xlabel, xmin, xmax, nbins, nbins_control in tqdm(variables):

    bins = np.linspace(xmin, xmax, nbins)

    # Plot config
    fig = plt.figure(figsize=(6,6))
    grid = [1,1] # number of rows, number of cols
    gspec = gs.GridSpec(grid[0], grid[1], width_ratios=[1], height_ratios=[1])

    # Plot code
    ax1 = plt.subplot(position(gspec, grid, main=1, sub=1)) # main is column number, sub is row number

    for signal in signal_to_use:
        ana.step_plot(
            ax1, var, ds[signal["key"]],
            label=signal["label"], color=signal["color"],
            weight="evtWeight", bins=bins
        )

    ybkg, errbkg = ana.stacked_plot(ax1, var, dataframes, labels, colors, weight="evtWeight", bins=bins)

    ana.labels(ax1, ylabel="Events", xlabel=xlabel)
    ana.style(
        ax1, lumi=metadata["luminosity"].get(dataset_year), year=year_style,
        ylog=True, legend_ncol=2, ylim=[1.e-2,1.e8]
    )

    plt.savefig(f"{plots_path}/regionID={region_id}/{var}.png", dpi=200, facecolor='white')
    plt.close()



