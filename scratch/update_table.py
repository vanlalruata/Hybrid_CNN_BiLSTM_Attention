import os

def main():
    tex_path = r"g:\PycharmProjects\PythonProject\HLC\manuscript\manuscript.tex"
    
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Replace the ANOVA sentence
    old_anova = "Kumar et al.~\\cite{kumar2021eddos} highlighted that neglecting statistical validation (such as ANOVA) alongside local and global explanations leads to surrogate explanations that are uncalibrated and difficult to verify by security analysts."
    new_anova = "Kumar et al.~\\cite{kumar2021eddos} proposed a distributed intrusion detection framework for blockchain-enabled fog environments, which achieved high detection rates but incurred significant resource overhead, highlighting the need for efficient edge-oriented models."
    
    if old_anova in content:
        content = content.replace(old_anova, new_anova)
        print("Successfully replaced the ANOVA sentence!")
    else:
        print("WARNING: ANOVA sentence not found exactly!")
        
    # Locate and replace the table
    # We want to replace the table environment from \begin{sidewaystable}[htbp] to \end{sidewaystable}
    start_tag = "\\begin{sidewaystable}[htbp]"
    end_tag = "\\end{sidewaystable}"
    
    start_idx = content.find(start_tag)
    end_idx = content.find(end_tag)
    
    if start_idx != -1 and end_idx != -1:
        end_idx += len(end_tag)
        
        # New table definition
        new_table = r"""\begin{sidewaystable}[htbp]
\centering
\caption{Comparison of related works in IoT intrusion and DDoS detection}
\label{tab:related_works_comparison}
\renewcommand{\arraystretch}{1.5}
\scriptsize
\begin{tabular}{lp{3.2cm}p{5.5cm}p{4.2cm}p{4.2cm}}
\toprule
Study & Proposed Method & Key Findings & Limitations & Research Gaps \\
\midrule
Verma and Ranga~\cite{verma2020machine} & Machine learning evaluation & Compares classifiers on IoT benchmarks & Single-node evaluation & Lacks multi-dataset deep learning benchmarks \\
Roopak et al.~\cite{roopak2019deep} & CNN + LSTM hybrid DL & High accuracy on DDoS subclass & Tested only on CICIDS2017 & No cross-dataset analysis or explainability \\
Kumar et al.~\cite{kumar2021eddos} & Deep Learning DDoS framework & Improved recall on IoT traffic & Heavy model architecture & High latency, unsuitable for real-time edge \\
Latif et al.~\cite{latif2020lightweight} & Lightweight Random Neural Network & Low prediction latency & Evaluated on older feature sets & Lacks deep spatial-temporal feature mapping \\
Song et al.~\cite{zhao2020hybrid} & CNN-LSTM model & Captured spatial-temporal structures & Evaluated on older datasets & No validation on modern IoT/IoMT environments \\
Jiang et al.~\cite{jiang2020network} & CNN-BiLSTM + Attention & Dynamic weight assignment & Complex, high parameter size & Missing inference latency/energy proxy checks \\
Junior et al.~\cite{zhou2021ddos} & Attention-LSTM model & High F1-score for DDoS classification & Captures only temporal structures & Ignores spatial/feature-level correlations \\
Swain et al.~\cite{wu2023lightweight} & Edge hybrid model & Balanced speed and accuracy & Restricted to single dataset & Gaps in explainability and model trust \\
Wang et al.~\cite{gu2022lightweight} & Attention-CNN-GRU & High throughput for benign/attack & Lacks robustness under class imbalance & Lacks global SHAP/local LIME validation \\
Halladay and Abbott~\cite{basnet2022deep} & Deep learning DDoS detection & Effective multiclass separation & Lacks feature harmonization & Cannot generalize to unseen IoT protocols \\
Capuano et al.~\cite{almimi2023xai} & XAI framework survey & Highlighted importance of model trust & Review only, no implementation & Lacks integration with hybrid DL models \\
Sarhan et al.~\cite{sarhan2020evaluation} & Feature set evaluation (NetFlow vs CIC) & NetFlow features improve generalizability and NIDS adaptiveness & Lacks integration of deep sequential architectures & Does not analyze cross-dataset performance with temporal models \\
Altunay and Albayrak~\cite{yan2023deep} & CNN+LSTM hybrid NIDS & Captures spatial-temporal IIoT traffic features with 99.8\% accuracy & No attention weighting or bidirectional temporal modeling & Lacks cross-dataset validation and model explainability \\
Hnamte et al.~\cite{hnamte2024iot} & Deep neural network DDoS detection & High accuracy (99.99\%) with XAI validation of predictions & Evaluated primarily under static SDN network topologies & Gaps in generalizing across highly heterogeneous IoT networks \\
Sgouras et al.~\cite{sgouras2022detecting} & ML DDoS impact analysis in smart grids & Quantified cyberattack impact on smart grid infrastructure stability & Heavy focus on impact metrics rather than real-time NIDS & Lacks modern multi-protocol feature validation \\
Zohourian et al.~\cite{hosseini2024attention} & IoT-PRIDS (Packet representation) & Leverages packet representation learning for robust NIDS & High feature extraction latency, unsuited for edge deployment & Gaps in global model explainability under class imbalance \\
Susilo and Sari~\cite{susilo2020security} & Deep learning algorithm NIDS in IoT & Explores MLP and LSTM structures for cyber anomaly detection & Severe performance degradation under cross-dataset traffic & Missing statistical feature selection (like ANOVA) and local XAI \\
Yin et al.~\cite{yin2020deep} & RNN-based NIDS & RNNs outperform feed-forward networks in modeling sequential traffic & Vanishing gradients during long sequence learning; high latency & Lacks spatial feature extraction and attention weight tuning \\
Memon et al.~\cite{shaaban2023biomt} & Explainable IDS for IoMT & High performance on healthcare devices with local SHAP interpretability & Evaluated on static features; ignores temporal relationships & Lacks bidirectional sequence modeling and feature harmonization \\
Mushtaq et al.~\cite{dong2021novel} & Autoencoder-LSTM hybrid & Two-stage model improves botnet detection accuracy via autoencoders & Spatial feature-level correlations are ignored; high training overhead & No validation on cross-dataset configurations or explainability \\
Moustafa et al.~\cite{liu2022explainable} & Explainable NIDS survey & Highlights opportunities for SHAP/LIME in NIDS transparency & Survey paper, lacks experimental hybrid DL model evaluation & Does not propose a unified framework for cross-dataset trust \\
Bhamare et al.~\cite{bhamare2020machine} & Supervised ML for cloud security & SVMs and Decision Trees achieve good anomaly classification rates & SVM is too computationally expensive for real-time edge nodes & Lacks deep sequential modeling and multi-dataset benchmarks \\
Churcher et al.~\cite{churcher2021machine} & Experimental evaluation of ML NIDS & Analyzed classifier sensitivity to feature noise and hardware constraints & Decision trees display low generalizability under traffic fluctuations & No deep learning or spatial-temporal feature fusion explored \\
Soe et al.~\cite{soe2020machine} & Ensemble ML with feature selection & Wrapper feature selection improves RF accuracy to 99\% & Evaluated on single dataset; sensitive to protocol drift & Lacks explainability and deep temporal learning \\
Min et al.~\cite{abbasi2021deep} & Memory-augmented autoencoder NIDS & Effectively models normal patterns to identify anomalies & High computational latency due to memory networks; unsuited for edge & No sequence modeling (like BiLSTM) or global explanations \\
Alshehri and Alnfiai~\cite{alomari2021hybrid} & Parallel hybrid DL DDoS detection & Parallel architectures capture features faster and with higher accuracy & Lacks feature harmonization, leading to poor generalization & Gaps in transparency; no local or global XAI models \\
Sarhan et al.~\cite{habib2023multi} & Multi-dataset standardization & Standardizing NetFlow features improves cross-dataset NIDS stability & Tested only on shallow classifiers; lacks deep sequence extraction & Gaps in attention refinement and deep learning integration \\
Proposed Work & Hybrid CNN--BiLSTM--Attention & Spatial-temporal fusion with XAI & Validated offline on traces & Physical deployment on live edge nodes \\
\bottomrule
\end{tabular}
\end{sidewaystable}"""
        
        content = content[:start_idx] + new_table + content[end_idx:]
        print("Successfully updated the table of related works!")
    else:
        print("ERROR: Could not locate table tags in manuscript.tex!")
        
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("Done!")

if __name__ == "__main__":
    main()
