import os
import shutil

replacements = {
    # 1. Related work and text sections
    "Verma and Ranga~\\cite{verma2020machine} concluded": "Verma and Ranga~\\cite{verma2020machine} evaluated machine learning configurations and concluded",
    "A systematic review of these machine learning configurations by Verma and Ranga~\\cite{verma2020machine}": "A comparative evaluation of these machine learning configurations by Verma and Ranga~\\cite{verma2020machine}",
    "Habib et al.~\\cite{habib2023multi}": "Sarhan et al.~\\cite{habib2023multi}",
    "Latif et al.~\\cite{latif2020lightweight} proposed a lightweight 1D-CNN architecture": "Latif et al.~\\cite{latif2020lightweight} proposed a lightweight random neural network",
    "Basnet et al.~\\cite{basnet2022deep}": "Halladay and Abbott~\\cite{basnet2022deep}",
    "Abbasi et al.~\\cite{abbasi2021deep}": "Min et al.~\\cite{abbasi2021deep}",
    "Yan and Han~\\cite{yan2023deep}": "Altunay and Albayrak~\\cite{yan2023deep}",
    "Zhao et al.~\\cite{zhao2020hybrid}": "Song et al.~\\cite{zhao2020hybrid}",
    "Dong and Abbott~\\cite{dong2021novel}": "Mushtaq et al.~\\cite{dong2021novel}",
    "designed a hybrid machine learning framework for botnet detection on BoT-IoT": "designed a hybrid machine learning framework for intrusion detection in vehicular networks based on ToN-IoT",
    "Al-Omari et al.~\\cite{alomari2021hybrid}": "Alshehri and Alnfiai~\\cite{alomari2021hybrid}",
    "Hosseini and Miri~\\cite{hosseini2024attention}": "Zohourian et al.~\\cite{hosseini2024attention}",
    "Zhou et al.~\\cite{zhou2021ddos}": "Junior et al.~\\cite{zhou2021ddos}",
    "Gu et al.~\\cite{gu2022lightweight}": "Wang et al.~\\cite{gu2022lightweight}",
    "Wu et al.~\\cite{wu2023lightweight}": "Swain et al.~\\cite{wu2023lightweight}",
    "Hnamte and Hussain~\\cite{hnamte2024iot}": "Hnamte et al.~\\cite{hnamte2024iot}",
    "Al-Mimi et al.~\\cite{almimi2023xai}": "Capuano et al.~\\cite{almimi2023xai}",
    "Shaaban and Abdelgawad~\\cite{shaaban2023biomt}": "Memon et al.~\\cite{shaaban2023biomt}",
    "Liu et al.~\\cite{liu2022explainable}": "Moustafa et al.~\\cite{liu2022explainable}",
    "Kumar et al.~\\cite{kumar2021eddos} highlighted that neglecting statistical validation (such as ANOVA) alongside local and global explanations leads to surrogate explanations that are uncalibrated and hard to verify by security analysts.": "Kumar et al.~\\cite{kumar2021eddos} proposed a distributed intrusion detection framework for blockchain-enabled fog environments, which achieved high detection rates but incurred significant resource overhead, highlighting the need for efficient edge-oriented models.",
    
    # 2. Table 3 comparison rows
    "Verma et al.~\\cite{verma2020machine} & Systematic review of NIDS & Outlines ML/DL algorithms in IoT NIDS & Qualitative, no new models & Lacks multi-dataset quantitative benchmarks \\\\": "Verma and Ranga~\\cite{verma2020machine} & Machine learning evaluation & Compares classifiers on IoT benchmarks & Single-node evaluation & Lacks multi-dataset deep learning benchmarks \\\\",
    "Latif et al.~\\cite{latif2020lightweight} & Lightweight 1D-CNN model & Low complexity for edge nodes & Lacks sequence context modeling & Ignores temporal traffic sequence structures \\\\": "Latif et al.~\\cite{latif2020lightweight} & Lightweight Random Neural Network & Low prediction latency & Evaluated on older feature sets & Lacks deep spatial-temporal feature mapping \\\\",
    "Zhao et al.~\\cite{zhao2020hybrid} & CNN-LSTM model & Captured spatial-temporal structures & Evaluated on older datasets & No validation on modern IoT/IoMT environments \\\\": "Song et al.~\\cite{zhao2020hybrid} & CNN-LSTM model & Captured spatial-temporal structures & Evaluated on older datasets & No validation on modern IoT/IoMT environments \\\\",
    "Zhou et al.~\\cite{zhou2021ddos} & Attention-LSTM model & High F1-score for DDoS classification & Captures only temporal structures & Ignores spatial/feature-level correlations \\\\": "Junior et al.~\\cite{zhou2021ddos} & Attention-LSTM model & High F1-score for DDoS classification & Captures only temporal structures & Ignores spatial/feature-level correlations \\\\",
    "Wu et al.~\\cite{wu2023lightweight} & Edge hybrid model & Balanced speed and accuracy & Restricted to single dataset & Gaps in explainability and model trust \\\\": "Swain et al.~\\cite{wu2023lightweight} & Edge hybrid model & Balanced speed and accuracy & Restricted to single dataset & Gaps in explainability and model trust \\\\",
    "Gu et al.~\\cite{gu2022lightweight} & Attention-CNN-GRU & High throughput for benign/attack & Lacks robustness under class imbalance & Lacks global SHAP/local LIME validation \\\\": "Wang et al.~\\cite{gu2022lightweight} & Attention-CNN-GRU & High throughput for benign/attack & Lacks robustness under class imbalance & Lacks global SHAP/local LIME validation \\\\",
    "Basnet et al.~\\cite{basnet2022deep} & Deep learning DDoS detection & Effective multiclass separation & Lacks feature harmonization & Cannot generalize to unseen IoT protocols \\\\": "Halladay and Abbott~\\cite{basnet2022deep} & Deep learning DDoS detection & Effective multiclass separation & Lacks feature harmonization & Cannot generalize to unseen IoT protocols \\\\",
    "Al-Mimi et al.~\\cite{almimi2023xai} & XAI framework survey & Highlighted importance of model trust & Review only, no implementation & Lacks integration with hybrid DL models \\\\": "Capuano et al.~\\cite{almimi2023xai} & XAI framework survey & Highlighted importance of model trust & Review only, no implementation & Lacks integration with hybrid DL models \\\\",

    # 3. SOTA comparison text and table
    "Pecorella et al.~\\cite{pecorella2021iot}": "Garcia et al.~\\cite{pecorella2021iot}",
    "Moustafa and Slay~\\cite{moustafa2015bot}": "Koroniotis et al.~\\cite{moustafa2015bot}",
    "Gharib et al.~\\cite{iomt2024dataset}": "Dadkhah et al.~\\cite{iomt2024dataset}",
    "Shaaban et al.~\\cite{shaaban2023biomt}": "Memon et al.~\\cite{shaaban2023biomt}",
    "Lashkari et al.~\\cite{ciciot2025dataset}": "Neto et al.~\\cite{ciciot2025dataset}",
    "Habib et al.~\\cite{habib2023multi}": "Sarhan et al.~\\cite{habib2023multi}",
    "Hnamte \\& Hussain~\\cite{hnamte2024iot}": "Hnamte et al.~\\cite{hnamte2024iot}",
    "Moustafa \\& Slay~\\cite{moustafa2015bot}": "Koroniotis et al.~\\cite{moustafa2015bot}",
    "Yan \\& Han~\\cite{yan2023deep}": "Altunay \\& Albayrak~\\cite{yan2023deep}",
}

def main():
    tex_path = r"g:\PycharmProjects\PythonProject\HLC\manuscript\manuscript.tex"
    backup_path = tex_path + ".bak"
    
    # Create backup
    shutil.copyfile(tex_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    replaced_count = 0
    for target, replacement in replacements.items():
        if target in content:
            content = content.replace(target, replacement)
            print(f"Replaced:\n  FROM: {target.strip()}\n  TO:   {replacement.strip()}")
            replaced_count += 1
        else:
            # Let's check with slightly flexible whitespace
            target_norm = " ".join(target.split())
            # Find in normalized content if it's there
            print(f"WARNING: Target not found exactly: '{target[:40]}...'")
            
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"Finished! Replaced {replaced_count} targets.")

if __name__ == "__main__":
    main()
