#Importar classes em diferentes diretórios

import sys
sys.path.append('/thiagosousa@analysis.acad.hepgrid/home/thiagosousa/IC_Helena/IC_Dark_Matter/hhdmAnalysis-IC/hhdm_analysis')
#sys.path.append('/thiagosousa@analysis.acad.hepgrid/home/thiagosousa') 
#sys.path.append('/home/thiagosousa/ANATools')


import anatools.data as data

import pprint
import json
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs


from utils import signal_label, stack_sorting, position, process_signals


import anatools.analysis as ana

ana.start()
plt.style.use("default")


# ------------ Configuração ------------

period = 'APV_16'
year_style = 2016
dataset_year = 'APV_2016'
basedir = '/home/thiagosousa/IC_Helena/Amostras/hhdmAnalysis_deepJet_Regions/datasets'
basedir2 = '/home/thiagosousa/IC_Helena/Amostras/hhdmAnalysis_deepCSV_SR/datasets'
region_id = 0 # Signal region

# Setup output folders
dataset_name = basedir.split('/')[-2]
plots_path = f"./figures/{dataset_name}/{dataset_year}"
available_region_ids = [0]
for available_region_id in available_region_ids:
    Path(f"{plots_path}/regionID={available_region_id}").mkdir(parents=True, exist_ok=True)
    
# Setup output folders
dataset_name2 = basedir2.split('/')[-2]
plots_path2 = f"./figures/{dataset_name2}/{dataset_year}"
available_region_ids = [0]
for available_region_id in available_region_ids:
    Path(f"{plots_path2}/regionID={available_region_id}").mkdir(parents=True, exist_ok=True)

#Cria o diretório Figures. Dentro dela mais 2 diretórios chamados: hhdmAnalysis_deepCSV_SR e hhdmAnalysis_deepJet_Regions
#onde ficarão os plots de cada ano dos dados respectivos de cada ano.



# ------------ Ler o Metadata ------------


with open('/home/thiagosousa/IC_Helena/IC_Dark_Matter/hhdmAnalysis-IC/metadata.json', 'r') as f:
    metadata = json.load(f)

ST = metadata.get("datasets").get("ST")
TT = metadata.get("datasets").get("TT")
ZZ = metadata.get("datasets").get("ZZ")
WZ = metadata.get("datasets").get("WZ")
DY = metadata.get("datasets").get("DY")
RESIDUAL = metadata.get("datasets").get("RESIDUAL")
DATA = metadata.get("datasets").get("DATA")



# ------------ Ler os Datasts ------------



variables = ["RegionID", "evtWeight", "LeadingJet_btag_score", "SubLeadingJet_btag_score"]

# Read data generated with DeepJet
ds_deepjet = data.read_files(basedir, period, mode="normal", features=variables)
data.join_datasets(ds_deepjet, "ST", ST.get(period), mode="normal")
data.join_datasets(ds_deepjet, "TT", TT.get(period), mode="normal")
data.join_datasets(ds_deepjet, "ZZ", ZZ.get(period), mode="normal")
data.join_datasets(ds_deepjet, "WZ", WZ.get(period), mode="normal")
data.join_datasets(ds_deepjet, "DYJetsToLL", DY.get(period), mode="normal")
data.join_datasets(ds_deepjet, "Residual", RESIDUAL.get(period), mode="normal")
data.join_datasets(ds_deepjet, "Data", DATA.get(period), mode="normal")

# Read data generated with DeepCSV
ds_deepcsv = data.read_files(basedir2, period, mode="normal", features=variables)
data.join_datasets(ds_deepcsv, "ST", ST.get(period), mode="normal")
data.join_datasets(ds_deepcsv, "TT", TT.get(period), mode="normal")
data.join_datasets(ds_deepcsv, "ZZ", ZZ.get(period), mode="normal")
data.join_datasets(ds_deepcsv, "WZ", WZ.get(period), mode="normal")
data.join_datasets(ds_deepcsv, "DYJetsToLL", DY.get(period), mode="normal")
data.join_datasets(ds_deepcsv, "Residual", RESIDUAL.get(period), mode="normal")
data.join_datasets(ds_deepcsv, "Data", DATA.get(period), mode="normal")




# ------------ Preparando os Plots ------------


# DeepJet
colors_list = ["gainsboro", "orchid", "limegreen", "red", "skyblue", "darkgoldenrod"]
labels_list = [r"Residual SM", r"$WZ$", r"$ZZ$", "Single top", r"$t\bar{t}$", "Drell-Yan"]
bkg_list = ["Residual", "WZ", "ZZ", "ST", "TT", "DYJetsToLL"]
ds_deepjet = {k: v[v.RegionID == region_id] for k,v in ds_deepjet.items()}
dataframes_deepjet, labels_deepjet, colors_deepjet = stack_sorting(ds_deepjet, colors_list, labels_list, bkg_list)

# DeepCSV
colors_list = ["gainsboro", "orchid", "limegreen", "red", "skyblue", "darkgoldenrod"]
labels_list = [r"Residual SM", r"$WZ$", r"$ZZ$", "Single top", r"$t\bar{t}$", "Drell-Yan"]
bkg_list = ["Residual", "WZ", "ZZ", "ST", "TT", "DYJetsToLL"]
ds_deepcsv = {k: v[v.RegionID == region_id] for k,v in ds_deepcsv.items()}
dataframes_deepcsv, labels_deepcsv, colors_deepcsv = stack_sorting(ds_deepcsv, colors_list, labels_list, bkg_list)




# ------------ Definindo as Variáveis do Plot ------------


# Signals to use in plot
signal_to_use = process_signals([
    ("Signal_400_100", "darkviolet"),
    ("Signal_1000_100", "blue")
])

# Variables to plot
variables = [
    ('LeadingJet_btag_score', 'leading jet btag score', 0, 1, 51, 5001),
    ('SubLeadingJet_btag_score', 'subleading jet btag score', 0, 1, 51, 5001),
]


# ------------ Plot da Distribuição DeepJet ------------


for var, xlabel, xmin, xmax, nbins, nbins_control in tqdm(variables):

    bins = np.linspace(xmin, xmax, nbins)
    bins_control = np.linspace(xmin, xmax, nbins_control)

    # Plot config
    fig = plt.figure(figsize=(7,10))
    grid = [2,1] # number of rows, number of cols
    gspec = gs.GridSpec(grid[0], grid[1], width_ratios=[1], height_ratios=[4,1])

    # Main plot
    ax1 = plt.subplot(position(gspec, grid, main=1, sub=1)) # main is column number, sub is row number
    for signal in signal_to_use:
        ana.step_plot(
            ax1, var, ds_deepjet[signal["key"]],
            label=signal["label"], color=signal["color"],
            weight="evtWeight", bins=bins
        )
        
    ybkg, errbkg = ana.stacked_plot(ax1, var, dataframes_deepjet, labels_deepjet, colors_deepjet, weight="evtWeight", bins=bins)
    btagging_values = metadata.get("btagging").get(dataset_year).get("DeepJet")
    btagging_colors = ["red", "green", "blue"]
    for idx, (wp_name, wp_value) in enumerate(btagging_values.items()):
        ax1.axvline(x=wp_value, label=wp_name, color=btagging_colors[idx], linestyle="dotted")
    
    ana.labels(ax1, ylabel="Events")
    ana.style(
        ax1, lumi=metadata["luminosity"].get(dataset_year), year=year_style,
        ylog=True, legend_ncol=2, ylim=[1.e-2,1.e8]
    )
    
    # Sub plot
    ax2 = plt.subplot(position(gspec, grid, main=1, sub=2)) # main is column number, sub is row number
    ctr = ana.control(var, [ds_deepjet[signal["key"]]], dataframes_deepjet, weight="evtWeight", bins=bins_control)
    ctr.signal_eff_plot(label=signal["label"])
    ctr.bkg_eff_plot(label="Background")
    ana.labels(ax2, xlabel=xlabel + " [DeepJet]", ylabel="Efficiency")
    ana.style(ax2, ylim=[0., 1.1], yticks=[0., 0.2, 0.4, 0.6, 0.8, 1.], xgrid=True, ygrid=True)

    btagging_values = metadata.get("btagging").get(dataset_year).get("DeepJet")
    btagging_colors = ["red", "green", "blue"]
    for idx, (wp_name, wp_value) in enumerate(btagging_values.items()):
        ax2.axvline(x=wp_value, label=wp_name, color=btagging_colors[idx], linestyle="dotted")
    
    # Save
    plt.savefig(f"{plots_path}/regionID={region_id}/{var}_deepJet.png", dpi=200, facecolor='white')
    plt.close()



# ------------ Plot da Distribuição DeepCSV ------------


for var, xlabel, xmin, xmax, nbins, nbins_control in tqdm(variables):

    bins = np.linspace(xmin, xmax, nbins)
    bins_control = np.linspace(xmin, xmax, nbins_control)

    # Plot config
    fig = plt.figure(figsize=(7,10))
    grid = [2,1] # number of rows, number of cols
    gspec = gs.GridSpec(grid[0], grid[1], width_ratios=[1], height_ratios=[4,1])

    # Main plot
    ax1 = plt.subplot(position(gspec, grid, main=1, sub=1)) # main is column number, sub is row number
    for signal in signal_to_use:
        ana.step_plot(
            ax1, var, ds_deepcsv[signal["key"]],
            label=signal["label"], color=signal["color"],
            weight="evtWeight", bins=bins
        )
        
    ybkg, errbkg = ana.stacked_plot(ax1, var, dataframes_deepcsv, labels_deepcsv, colors_deepcsv, weight="evtWeight", bins=bins)
    btagging_values = metadata.get("btagging").get(dataset_year).get("DeepCSV")
    btagging_colors = ["red", "green", "blue"]
    for idx, (wp_name, wp_value) in enumerate(btagging_values.items()):
        ax1.axvline(x=wp_value, label=wp_name, color=btagging_colors[idx], linestyle="dotted")
    
    ana.labels(ax1, ylabel="Events")
    ana.style(
        ax1, lumi=metadata["luminosity"].get(dataset_year), year=year_style,
        ylog=True, legend_ncol=2, ylim=[1.e-2,1.e8]
    )
    
    # Sub plot
    ax2 = plt.subplot(position(gspec, grid, main=1, sub=2)) # main is column number, sub is row number
    ctr = ana.control(var, [ds_deepcsv[signal["key"]]], dataframes_deepcsv, weight="evtWeight", bins=bins_control)
    ctr.signal_eff_plot(label=signal["label"])
    ctr.bkg_eff_plot(label="Background")
    ana.labels(ax2, xlabel=xlabel + " [DeepCSV]", ylabel="Efficiency")
    ana.style(ax2, ylim=[0., 1.1], yticks=[0., 0.2, 0.4, 0.6, 0.8, 1.], xgrid=True, ygrid=True)

    btagging_values = metadata.get("btagging").get(dataset_year).get("DeepCSV")
    btagging_colors = ["red", "green", "blue"]
    for idx, (wp_name, wp_value) in enumerate(btagging_values.items()):
        ax2.axvline(x=wp_value, label=wp_name, color=btagging_colors[idx], linestyle="dotted")
    
    # Save
    plt.savefig(f"{plots_path2}/regionID={region_id}/{var}_deepCSV.png", dpi=200, facecolor='white')
    plt.close()


