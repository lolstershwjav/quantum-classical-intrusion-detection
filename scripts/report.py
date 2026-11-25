"""Plot measured performance and circuit gates and render technical report."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

ROOT=Path("results")
LABELS={"linear_svm":"Linear SVM","rbf_svm":"RBF SVM","mlp":"Classical MLP",
        "qkernel_angle":"Angle kernel","qkernel_phase":"Phase kernel","vqc":"VQC","hybrid":"Hybrid"}
COLORS=("#426588","#7c629e","#2a806f","#b56c40")


def circuit():
    fig, ax=plt.subplots(figsize=(8.8,2.4))
    ax.set(xlim=(0,8.6),ylim=(-.6,3.7))
    ax.axis("off")
    for q in range(4):
        y=3-q
        ax.hlines(y,.6,8.3,color="#8196a7",linewidth=1)
        ax.text(.1,y,f"q{q}",va="center",weight="semibold")
        for x, label in [(1.45,"RY(x + θ)"),(2.8,"RZ(φ)")]:
            patch=FancyBboxPatch((x-.49,y-.25),.98,.5,boxstyle="round,pad=.02",
                                 ec="#426588",fc="#e8f2f9",lw=.8,zorder=3)
            ax.add_patch(patch)
            ax.text(x,y,label,va="center",ha="center",fontsize=9,zorder=4)
    for q in range(4):
        x=4.0+.95*q
        a,b=3-q,3-((q+1)%4)
        ax.plot([x,x],[a,b],color="#19765f",linewidth=1.7)
        ax.plot(x,a,"o",color="#19765f",ms=6)
        ax.plot(x,b,"o",color="#19765f",mfc="white",ms=13)
        ax.text(x,b,"+",ha="center",va="center",fontsize=12,color="#19765f")
    fig.savefig(ROOT/"circuit.png",dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def charts(index):
    fig,axes=plt.subplots(1,2,figsize=(10.3,3.35))
    models=list(LABELS)
    for q, color, dx in [(4,COLORS[0],-.16),(6,COLORS[2],.16)]:
        values=[index[q,384,name,"ideal"]["ap"] for name in models]
        mean=[v["mean"]*100 for v in values]
        axes[0].bar(np.arange(len(models))+dx,mean,.29,label=f"{q} features",color=color)
    axes[0].set(xticks=np.arange(len(models)),xticklabels=[LABELS[m] for m in models],
                ylabel="Average precision (%)",ylim=(0,85), title="Ideal readout · 384 fit rows")
    axes[0].tick_params(axis="x",rotation=35)
    axes[0].legend(frameon=False,fontsize=8)
    conditions=["ideal","shots1024","shots256","noisy256"]
    for model,color in zip(("qkernel_angle","qkernel_phase","vqc","hybrid"),COLORS):
        vals=[100*index[6,384,model,c]["ap"]["mean"] for c in conditions]
        axes[1].plot(range(4),vals,"o-",ms=4,label=LABELS[model],color=color)
    axes[1].axhline(100*index[6,384,"rbf_svm","ideal"]["ap"]["mean"],linestyle="--",
                    color="#647582",label="RBF SVM")
    axes[1].set(xticks=range(4),xticklabels=["Ideal","1,024 shots","256 shots","256 + noise"],
                ylim=(58,77),title="Six qubits · 384 fit rows")
    axes[1].legend(frameon=False,fontsize=8,ncol=2)
    for ax in axes:
        ax.grid(axis="y",alpha=.24)
        ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(w_pad=2)
    fig.savefig(ROOT/"comparisons.png",dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def pdf(data,index):
    dark=colors.HexColor("#18394d")
    folder=Path(matplotlib.get_data_path())/"fonts"/"ttf"
    pdfmetrics.registerFont(TTFont("StudySans",str(folder/"DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("StudySans-Bold",str(folder/"DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("StudySans",normal="StudySans",bold="StudySans-Bold")
    styles=getSampleStyleSheet()
    for k in ("Normal","BodyText"):
        styles[k].fontName="StudySans"
    for k in ("Title","Heading2"):
        styles[k].fontName="StudySans-Bold"
        styles[k].textColor=dark
    styles["Title"].alignment=0
    styles["Title"].fontSize=24
    styles["Title"].leading=28
    styles["BodyText"].leading=14
    styles["BodyText"].spaceAfter=8
    body=[]
    def p(s,key="BodyText"):
        body.append(Paragraph(s,styles[key]))
    def section(title,txt):
        p(title,"Heading2")
        p(txt)
    def graphic(file,height):
        body.append(Image(str(ROOT/file),width=480,height=height,kind="proportional"))

    p("Quantum vs. Classical Intrusion Detection","Title")
    p("Matched feature, sample-size and finite-shot comparisons on UNSW-NB15")
    p(f"<b>Abstract.</b> Across {len(data['runs'])} measurements, we compare linear and RBF SVMs, a "
      "classical MLP, two fidelity feature maps, a variational circuit and a hybrid head. "
      "The ideal six-qubit phase kernel yields 72.51% average precision versus 71.02% for the "
      "classical RBF. Its paired 95% bootstrap interval spans -0.73 to +4.00 points, "
      "and finite-shot noisy readout yields 69.81%.")
    section("1. Research question and related work",
            "Do nonlinear quantum feature maps and variational circuits improve intrusion "
            "classification after accounting for classical input representations? Fidelity "
            "kernels estimate feature-space similarity [1, 2]. Earlier research emphasizes "
            "classical controls and access to the same data when comparing model families [3].")
    section("2. Data and methodology",
            "Select distinct 16-dimensional UNSW-NB15 flow features [4]. Use separate "
            "training, validation and test draws. Train-only PCA yields four or six features. "
            "Training sizes are 192 and 384. All model comparisons within a seed share "
            "the feature representation. The test set is balanced and has 384 rows.")
    section("3. Circuit design and optimization",
            "Angle encoding applies RY rotations. The phase map adds commuting Z and ZZ "
            "phases plus a final RY layer. The simulator fuses commuting phases, with "
            "a direct gate-equivalence test. The variational model uses two blocks of "
            "RY, RZ and a directed CNOT ring, with either a single Z readout or an augmented "
            "classical head reading all Z expectations and input angles.")
    graphic("circuit.png",147)
    p("<b>Figure 1.</b> One four-wire variational block. The experiment repeats the block twice.")
    body.append(PageBreak())
    section("4. Quantitative results",
            "Table 1 reports six-feature means at 384 training rows. Noisy conditions use "
            "256 shots and 8% uniform outcome mixing. Fit time includes model fitting "
            "and, for quantum kernels, reference-state generation and Gram processing.")
    rows=[["Model","Ideal AP (%)","Noisy AP (%)","Ideal fit (ms)"]]
    for m,label in LABELS.items():
        a=index[6,384,m,"ideal"]
        noise=index.get((6,384,m,"noisy256"))
        rows.append([label,f"{100*a['ap']['mean']:.2f}",
                     f"{100*noise['ap']['mean']:.2f}" if noise else "—",
                     f"{1000*a['fit_seconds']['mean']:.1f}"])
    table=Table(rows,colWidths=[162,95,96,124],hAlign="LEFT")
    table.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"StudySans"),
         ("FONTNAME",(0,0),(-1,0),"StudySans-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
         ("BACKGROUND",(0,0),(-1,0),dark),("TEXTCOLOR",(0,0),(-1,0),colors.white),
         ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eff5f7")]),
         ("ALIGN",(1,0),(-1,-1),"RIGHT"),("TOPPADDING",(0,0),(-1,-1),5),
         ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    body.append(table)
    graphic("comparisons.png",177)
    p("<b>Figure 2.</b> Ideal ranking and readout-condition sensitivity.")
    section("5. Sample efficiency and scale",
            "At 192 rows and four features, the ideal phase-kernel mean AP difference "
            "relative to the RBF was +5.26 points. It was -0.43 points with six features. "
            "Controlled results differ by feature dimension and sample budget. The "
            "statevector scales as 2ⁿ and overlap evaluations scale quadratically with "
            "training rows. Neither simulation time nor these accuracy results establish a "
            "hardware advantage.")
    section("6. Limits and follow-up",
            "A single balanced cyber-range corpus and small samples limit inference. Future "
            "comparisons should test external prevalence, alternative feature maps, "
            "hardware noise and classical models on expanded features. Full gates, "
            "measurements and kernel repair details accompany the source.")
    p("References","Heading2")
    p("[1] Havlíček et al., Nature (2019). [2] Schuld and Killoran, Physical Review Letters "
      "(2019). [3] Huang et al., Nature Communications (2021). [4] Moustafa and Slay, "
      "MilCIS (2015). Full references appear in docs/METHODOLOGY.md.")
    def footer(c,doc):
        c.setFont("StudySans",8)
        c.setFillColor(dark)
        c.drawString(52,26,"Quantum vs. Classical Intrusion Detection")
        c.drawRightString(553,26,str(doc.page))
    doc=SimpleDocTemplate(str(ROOT/"quantum-intrusion-detection-report.pdf"),
                          title="Quantum vs. Classical Intrusion Detection",
                          leftMargin=52,rightMargin=52,topMargin=42,bottomMargin=42)
    doc.build(body,onFirstPage=footer,onLaterPages=footer)


if __name__=="__main__":
    with gzip.open(ROOT/"study.json.gz","rt",encoding="utf8") as source:
        data=json.load(source)
    index={(r["qubits"],r["train_size"],r["model"],r["condition"]):r["metrics"]
           for r in data["summary"]}
    circuit()
    charts(index)
    pdf(data,index)
