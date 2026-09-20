# Conceptual Model

Crecerelle analyses single-cell transcriptomic signatures from different facets of transcriptomic regulation:

- variation in gene expression (GE), represented as one abundance profile per cell,
- alternative splicing-induced transcript usage (TU), representing splicing-derived isoform or isoform usage patterns.

The package and notebooks use three modelling tracks:

- `scVI` for gene expression-only cell embeddings,
- `tuVI` for alternative splicing-induced transcript usage-only cell embeddings,
- `TRVI` for joint gene expression and alternative splicing-induced transcript usage cell embeddings.

The paper describes the motivation as moving beyond gene-expression-only interpretation by modelling transcript usage as a complementary transcriptomic layer. The README describes the downstream use of learnt GE, TU, and GE-TU embeddings for cell annotation, integration of gene expression and transcript usage, and functional characterisation.

At a workflow level, the notebooks:

1. preprocess GE and TU AnnData objects,
2. train or load model checkpoints,
3. evaluate cell embeddings and model variants,
4. run inference and downstream analyses,
5. save figures, CSV summaries, AnnData files, and model checkpoints.
