# Crecerelle
Crecerelle is a suite of deep generative models enabling cell development analy-
sis based on single-cell gene expression and alternative splicing. Pre-processing and quality
controls of scRNA-seq yield a gene expression (GE) matrix and transcript usage (TU) matrix. Cell
embeddings based solely on gene expression or transcript usage can then be learnt through scVI ([2])
and scTUVI respectively. Joint cell embeddings of both gene expression and splicing are inferred
through scGETUVI. The learnt GE, TU, or GE-TU cell embeddings are then applied for cell anno-
tation, gene expression and transcript usage integration, and functional characterisation.
