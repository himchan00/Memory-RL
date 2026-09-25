"""Shared matplotlib style for every paper-figure notebook.

Font sizes live HERE only; the figure notebooks (main_figure_vis, return_graph_vis, final_return_vs_depth_vis,
hidden_size_ablation_vis, seq_len_ablation_vis, check_time_vis) and render_depth_panels.py all call
`set_paper_style()` from this module. Their margin helpers read the sizes back from rcParams, so the layouts
follow automatically. After editing, restart the notebook kernel (or `importlib.reload(paper_style)`).
"""
import matplotlib

# ICLR 2027: 10 pt Times body, 5.5 in text width. Figures are included at native size, so these are final sizes.
TITLE_PT = 9        # panel / column titles (bold)
LABEL_PT = 8        # x / y axis labels
LEGEND_PT = 8
TICK_PT = 7         # tick labels; also the floor for any annotation text


def set_paper_style():
    # Times-metric fonts available on this box: Nimbus Roman / Liberation Serif / STIXGeneral;
    # "Times New Roman" is used when present.
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": LABEL_PT,
        "axes.labelsize": LABEL_PT,
        "axes.titlesize": TITLE_PT,
        "axes.titleweight": "bold",
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "legend.fontsize": LEGEND_PT,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "axes.labelpad": 1.5,
        "axes.titlepad": 3.0,
        "grid.linewidth": 0.4,
        "savefig.dpi": 300,
    })
