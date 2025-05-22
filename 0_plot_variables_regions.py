# Importe das Bibliotecas

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
basedir = "/home/thiagosousa/IC_Helena/Amostras/hhdmAnalysis_deepJet_Regions/datasets"
region_id = 0

# Setup output folders
dataset_name = basedir.split('/')[-2]
plots_path = f"./figures/{dataset_name}/{dataset_year}"
available_region_ids = [0,1,2,3,4]
for available_region_id in available_region_ids:
    Path(f"{plots_path}/regionID={available_region_id}").mkdir(parents=True, exist_ok=True)



# Lendo Metadata




with open('/home/thiagosousa/IC_Helena/IC_Dark_Matter/hhdmAnalysis-IC/metadata.json', 'r') as f:
    metadata = json.load(f)

ST = metadata.get("datasets").get("ST")
TT = metadata.get("datasets").get("TT")
ZZ = metadata.get("datasets").get("ZZ")
WZ = metadata.get("datasets").get("WZ")
DY = metadata.get("datasets").get("DY")
RESIDUAL = metadata.get("datasets").get("RESIDUAL")
DATA = metadata.get("datasets").get("DATA")


# Carregando os Datasets


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
ds = {k: v[v.RegionID == region_id] for k,v in ds.items()}
dataframes, labels, colors = stack_sorting(ds, colors_list, labels_list, bkg_list)


# Definindo variáveis para o Plot


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



# Plot da Distribuição


if region_id in [0]:
    ylim = [1.e-2,1.e6]
elif region_id in [1, 2]:
    ylim = [1.e-2,1.e8]
elif region_id in [3, 4]:
    ylim = [1.e-2,1.e4]

for var, xlabel, xmin, xmax, nbins, nbins_control in tqdm(variables):

    bins = np.linspace(xmin, xmax, nbins)
    
    if region_id == 0:

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
            ylog=True, legend_ncol=2, ylim=ylim
        )
        
    else:
        
        # Skip plotting Nbjets in TTBar control region since the is a dedicated notebook for this
        if region_id == 2 and var == "Nbjets":
            continue

        # Plot config
        fig = plt.figure(figsize=(6,7.5))
        grid = [2,1] # number of rows, number of cols
        gspec = gs.GridSpec(grid[0], grid[1], width_ratios=[1], height_ratios=[4, 1])

        # Plot code
        ax1 = plt.subplot(position(gspec, grid, main=1, sub=1)) # main is column number, sub is row number

        for signal in signal_to_use:
            ana.step_plot(
                ax1, var, ds[signal["key"]],
                label=signal["label"], color=signal["color"],
                weight="evtWeight", bins=bins
            )

        ybkg, errbkg = ana.stacked_plot(ax1, var, dataframes, labels, colors, weight="evtWeight", bins=bins)
        ydata, errdata = ana.data_plot(ax1, var, ds["Data"], bins=bins)

        ana.labels(ax1, ylabel="Events")
        ana.style(
            ax1, lumi=metadata["luminosity"].get(dataset_year), year=year_style,
            ylog=True, legend_ncol=2, ylim=ylim, xticklabels=False
        )
        
        # Sub plot
        ax2 = plt.subplot(position(gspec, grid, main=1, sub=2)) # main is column number, sub is row number
        ana.ratio_plot( ax2, ydata, errdata, ybkg, errbkg, bins=bins)
        ana.labels(ax2, xlabel=xlabel, ylabel="Data / Bkg.")
        ana.style(ax2, ylim=[0., 2], yticks=[0, 0.5, 1, 1.5, 2], xgrid=True, ygrid=True)

    plt.savefig(f"{plots_path}/regionID={region_id}/{var}.png", dpi=200, facecolor='white')
    plt.close()



# Plot da ROC Curve


if region_id == 0:

    for var, xlabel, xmin, xmax, nbins, nbins_control in tqdm(variables):

        bins_control = np.linspace(xmin, xmax, nbins_control)

        # Plot config
        fig = plt.figure(figsize=(7,7))
        grid = [1,1] # number of rows, number of cols
        gspec = gs.GridSpec(grid[0], grid[1], width_ratios=[1], height_ratios=[1])

        # Plot code
        ax1 = plt.subplot(position(gspec, grid, main=1, sub=1)) # main is column number, sub is row number

        for signal in signal_to_use:
            ctr = ana.control(var, [ds[signal["key"]]], dataframes, weight="evtWeight", bins=bins_control)
            ctr.roc_plot(label=signal["label"], color=signal["color"])

        ana.labels(ax1, ylabel="Signal Efficiency", xlabel=f"Background rejection [{xlabel}]")
        ana.style(
            ax1, lumi=metadata["luminosity"].get(dataset_year), year=year_style,
            ygrid=True, xgrid=True, legend_ncol=2, ylim=[0., 1.]
        )
        plt.savefig(f"{plots_path}/regionID={region_id}/{var}-ROC.png", dpi=200, facecolor='white')
        plt.close()
        
else:
    print("Not necessary to plot unless in Signal Region")



# Jet deltaEta in SR1 and SR2

'''
--> SR1: Nbjets == 1 and Njets >= 2
--> SR2: Nbjets >= 2
'''


if region_id == 0:
    ds_filtered = {k: v[["RegionID", "evtWeight", "Dijet_deltaEta", "Nbjets", "Njets"]] for k,v in ds.items()}

    # Prepare SR1 data
    ds_filtered_sr1 = {k: v[(v.Nbjets == 1) & (v.Njets >= 2)] for k,v in ds_filtered.items()}
    colors_list = ["gainsboro", "orchid", "limegreen", "red", "skyblue", "darkgoldenrod"]
    labels_list = [r"Residual SM", r"$WZ$", r"$ZZ$", "Single top", r"$t\bar{t}$", "Drell-Yan"]
    bkg_list = ["Residual", "WZ", "ZZ", "ST", "TT", "DYJetsToLL"]
    ds_filtered_sr1 = {k: v[v.RegionID == region_id] for k,v in ds_filtered_sr1.items()}
    dataframes_sr1, labels_sr1, colors_sr1 = stack_sorting(ds_filtered_sr1, colors_list, labels_list, bkg_list)
    
    # Prepare SR2 data
    ds_filtered_sr2 = {k: v[v.Nbjets >= 2] for k,v in ds_filtered.items()}
    colors_list = ["gainsboro", "orchid", "limegreen", "red", "skyblue", "darkgoldenrod"]
    labels_list = [r"Residual SM", r"$WZ$", r"$ZZ$", "Single top", r"$t\bar{t}$", "Drell-Yan"]
    bkg_list = ["Residual", "WZ", "ZZ", "ST", "TT", "DYJetsToLL"]
    ds_filtered_sr2 = {k: v[v.RegionID == region_id] for k,v in ds_filtered_sr2.items()}
    dataframes_sr2, labels_sr2, colors_sr2 = stack_sorting(ds_filtered_sr2, colors_list, labels_list, bkg_list)

else:
    print("Not necessary to plot unless in Signal Region")




if region_id == 0:
    # Signals to use in plot
    signal_to_use = process_signals([
        ("Signal_400_100", "darkviolet"),
        ("Signal_1000_100", "blue")
    ])

    var = "Dijet_deltaEta"
    xmin = 0
    xmax = 6
    nbins = 51
    xlabel = r"Dijet $\Delta \eta$"
    bins = np.linspace(xmin, xmax, nbins)

    # Plot config
    fig = plt.figure(figsize=(6,6))
    grid = [1,1] # number of rows, number of cols
    gspec = gs.GridSpec(grid[0], grid[1], width_ratios=[1], height_ratios=[1])

    # Plot code
    ax1 = plt.subplot(position(gspec, grid, main=1, sub=1)) # main is column number, sub is row number

    for signal in signal_to_use:
        ana.step_plot(
            ax1, var, ds_filtered_sr1[signal["key"]],
            label=signal["label"], color=signal["color"],
            weight="evtWeight", bins=bins
        )

    ybkg, errbkg = ana.stacked_plot(ax1, var, dataframes_sr1, labels_sr1, colors_sr1, weight="evtWeight", bins=bins)

    ana.labels(ax1, ylabel="Events", xlabel=xlabel)
    ana.style(
        ax1, lumi=metadata["luminosity"].get(dataset_year), year=year_style,
        ylog=True, legend_ncol=2, ylim=[1.e-2,1.e6]
    )
    plt.savefig(f"{plots_path}/regionID={region_id}/{var}-SR1.png", dpi=200, facecolor='white')
    plt.show()
    plt.close()

else:
    print("Not necessary to plot unless in Signal Region")





if region_id == 0:
    # Signals to use in plot
    signal_to_use = process_signals([
        ("Signal_400_100", "darkviolet"),
        ("Signal_1000_100", "blue")
    ])

    var = "Dijet_deltaEta"
    xmin = 0
    xmax = 6
    nbins = 51
    xlabel = r"Dijet $\Delta \eta$"
    bins = np.linspace(xmin, xmax, nbins)

    # Plot config
    fig = plt.figure(figsize=(6,6))
    grid = [1,1] # number of rows, number of cols
    gspec = gs.GridSpec(grid[0], grid[1], width_ratios=[1], height_ratios=[1])

    # Plot code
    ax1 = plt.subplot(position(gspec, grid, main=1, sub=1)) # main is column number, sub is row number

    for signal in signal_to_use:
        ana.step_plot(
            ax1, var, ds_filtered_sr2[signal["key"]],
            label=signal["label"], color=signal["color"],
            weight="evtWeight", bins=bins
        )

    ybkg, errbkg = ana.stacked_plot(ax1, var, dataframes_sr2, labels_sr2, colors_sr2, weight="evtWeight", bins=bins)

    ana.labels(ax1, ylabel="Events", xlabel=xlabel)
    ana.style(
        ax1, lumi=metadata["luminosity"].get(dataset_year), year=year_style,
        ylog=True, legend_ncol=2, ylim=[1.e-2,1.e6]
    )
    plt.savefig(f"{plots_path}/regionID={region_id}/{var}-SR2.png", dpi=200, facecolor='white')
    plt.show()
    plt.close()

else:
    print("Not necessary to plot unless in Signal Region")









