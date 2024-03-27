# DeepEpiIL13: Deep Learning for Rapid and Accurate Prediction of IL-13 Inducing Epitopes using Pre-trained Language Models and Multi-Window Convolutional Neural Networks

## Abstract <a name="abstract"></a>
Interleukin-13 (IL-13) is a key cytokine involved in allergic inflammation and the cytokine storm associated with severe COVID-19. Identifying antigenic epitopes capable of inducing IL-13 holds therapeutic potential for developing epitope-based vaccines and immunomodulatory treatments. However, conventional methods for epitope prediction are often inefficient and inaccurate. This study presents DeepEpiIL13, a novel deep learning framework that integrates pre-trained language models and multi-window convolutional neural networks (CNNs) to rapidly and accurately predict IL-13-inducing epitopes from protein sequences.

The ProtTrans pre-trained language model was employed to generate high-dimensional embeddings that capture rich contextual information from protein sequences. These embeddings were then inputted into a multi-window CNN architecture, enabling the efficient exploration of local and global sequence patterns associated with IL-13 induction. The proposed approach was rigorously evaluated using benchmark datasets and an independent SARS-CoV-2 dataset.

DeepEpiIL13 achieved superior performance compared to traditional methods, with an impressive Matthews correlation coefficient (MCC) of 0.52 and an area under the receiver operating characteristic curve (AUC) of 0.86 on the benchmark dataset. On the independent SARS-CoV-2 dataset, DeepEpiIL13 demonstrated remarkable robustness, attaining an MCC of 0.63 and an AUC of 0.92, highlighting its potential applicability in the context of COVID-19 research.

This study introduces a powerful deep learning framework for accurate epitope prediction, offering new avenues for the development of epitope-based vaccines and immunotherapies targeting IL-13-mediated disorders. The successful identification of IL-13-inducing epitopes paves the way for novel therapeutic interventions against allergic diseases, inflammatory conditions, and potentially severe viral infections like COVID-19. 
<br>
![workflow](Figure/Workflow.png)
