TABULA_MURIS_CELL_TYPES = 'B cell', 'Bergmann glial cell', 'Brush cell of epithelium proper of large intestine', \
                          'CD4-positive, alpha-beta T cell', 'CD8-positive, alpha-beta T cell', 'DN4 thymocyte', \
                          'Kupffer cell', 'NK cell', 'T cell', 'adventitial cell', 'aortic endothelial cell', \
                          'astrocyte', 'atrial myocyte', 'basal cell', 'basal cell of epidermis', \
                          'basal epithelial cell of tracheobronchial tree', 'basophil', 'bladder cell', \
                          'bladder urothelial cell', 'brain pericyte', 'bronchial smooth muscle cell', \
                          'bulge keratinocyte', 'chondrocyte', 'ciliated columnar cell of tracheobronchial tree', \
                          'classical monocyte', 'club cell of bronchiole', 'dendritic cell', 'early pro-B cell', \
                          'endocardial cell', 'endothelial cell', 'endothelial cell of coronary artery', \
                          'endothelial cell of hepatic sinusoid', 'endothelial cell of lymphatic vessel', \
                          'enterocyte of epithelium of large intestine', 'enteroendocrine cell', 'ependymal cell', \
                          'epidermal cell', 'epithelial cell', 'epithelial cell of large intestine', \
                          'epithelial cell of proximal tubule', 'epithelial cell of thymus', 'fenestrated cell', \
                          'fibroblast', 'fibroblast of cardiac tissue', 'fibroblast of lung', 'fibrocyte', \
                          'granulocyte', 'granulocyte monocyte progenitor cell', 'granulocytopoietic cell', \
                          'hematopoietic stem cell', 'hepatocyte', 'immature B cell', 'intermediate monocyte', \
                          'interneuron', 'intestinal crypt stem cell', 'keratinocyte', 'keratinocyte stem cell', \
                          'kidney collecting duct epithelial cell', 'kidney collecting duct principal cell', \
                          'kidney loop of Henle ascending limb epithelial cell', 'large intestine goblet cell', \
                          'late pro-B cell', 'leukocyte', 'luminal epithelial cell of mammary gland', \
                          'lung macrophage', 'lung neuroendocrine cell', 'lymphocyte', 'macrophage', \
                          'mature NK T cell', 'mature alpha-beta T cell', 'medium spiny neuron', \
                          'megakaryocyte-erythroid progenitor cell', 'mesangial cell', 'mesenchymal stem cell', \
                          'mesenchymal stem cell of adipose', 'microglial cell', 'monocyte', 'mucus secreting cell', \
                          'myeloid cell', 'myeloid dendritic cell', 'myeloid leukocyte', 'naive B cell', \
                          'neuroepithelial cell', 'neuron', 'neuronal stem cell', 'neutrophil', \
                          'non-classical monocyte', 'oligodendrocyte', 'oligodendrocyte precursor cell', \
                          'pancreatic A cell', 'pancreatic B cell', 'pancreatic D cell', 'pancreatic PP cell', \
                          'pancreatic acinar cell', 'pancreatic ductal cell', 'pancreatic stellate cell', \
                          'pericyte cell', 'plasmacytoid dendritic cell', 'precursor B cell', 'proerythroblast', \
                          'professional antigen presenting cell', 'promonocyte', 'pulmonary interstitial fibroblast', \
                          'regulatory T cell', 'respiratory basal cell', 'secretory cell', \
                          'skeletal muscle satellite cell', 'smooth muscle cell', \
                          'smooth muscle cell of the pulmonary artery', 'smooth muscle cell of trachea', \
                          'stromal cell', 'thymocyte', 'type I pneumocyte', 'type II pneumocyte', \
                          'valve cell', 'vein endothelial cell', 'ventricular myocyte'

TABULA_MURIS_TISSUE_CELL_DICTIONARY = {
    'Lung': ['B cell','CD4-positive, alpha-beta T cell','CD8-positive, alpha-beta T cell','NK cell','adventitial cell',
        'bronchial smooth muscle cell','ciliated columnar cell of tracheobronchial tree','classical monocyte',
        'club cell of bronchiole','dendritic cell','endothelial cell of lymphatic vessel','fibroblast of lung',
        'intermediate monocyte','leukocyte','lung macrophage','lung neuroendocrine cell','lymphocyte',
        'myeloid dendritic cell','neutrophil','non-classical monocyte','pericyte cell','plasmacytoid dendritic cell',
        'pulmonary interstitial fibroblast','regulatory T cell','respiratory basal cell',
        'smooth muscle cell of the pulmonary artery','type I pneumocyte','type II pneumocyte','vein endothelial cell'
    ],
    'Bladder': [
        'bladder cell', 'bladder urothelial cell'
    ],
    'Brain_Non-Myeloid': [
        'Bergmann glial cell','CD8-positive, alpha-beta T cell','T cell','astrocyte','brain pericyte',
        'endothelial cell','ependymal cell','interneuron','mature NK T cell','medium spiny neuron','neuron',
        'neuronal stem cell','oligodendrocyte','oligodendrocyte precursor cell'
    ],
    'Tongue': [
        'basal cell of epidermis', 'keratinocyte'
    ],
    'Brain_Myeloid': [
        'macrophage', 'microglial cell'
    ],
    'Marrow': [
        'CD4-positive, alpha-beta T cell','NK cell','basophil','early pro-B cell','granulocyte',
        'granulocyte monocyte progenitor cell','granulocytopoietic cell','hematopoietic stem cell','immature B cell',
        'late pro-B cell','macrophage','mature alpha-beta T cell','megakaryocyte-erythroid progenitor cell',
        'naive B cell','precursor B cell','promonocyte'
    ],
    'Skin': [
        'T cell','basal cell of epidermis','bulge keratinocyte','epidermal cell','keratinocyte stem cell','macrophage'
    ],
    'Heart': [
        'B cell','T cell','atrial myocyte','endocardial cell','endothelial cell of coronary artery',
        'fibroblast of cardiac tissue','monocyte','smooth muscle cell','valve cell'
    ],
    'SCAT': [
        'B cell','CD4-positive, alpha-beta T cell','CD8-positive, alpha-beta T cell','NK cell','T cell',
        'endothelial cell','epithelial cell','mesenchymal stem cell of adipose','myeloid cell'
    ],
    'GAT': [
        'B cell','CD4-positive, alpha-beta T cell','CD8-positive, alpha-beta T cell','NK cell','T cell',
        'endothelial cell','epithelial cell','mesenchymal stem cell of adipose','myeloid cell'
    ],
    'Trachea': [
        'T cell','basal epithelial cell of tracheobronchial tree','chondrocyte',
        'ciliated columnar cell of tracheobronchial tree','endothelial cell','fibroblast','granulocyte','macrophage',
        'mucus secreting cell','smooth muscle cell of trachea'
    ],
    'Diaphragm': [
        'B cell','T cell','endothelial cell','macrophage','mesenchymal stem cell','skeletal muscle satellite cell'
    ],
    'Kidney': [
        'B cell','T cell','epithelial cell of proximal tubule','fenestrated cell',
        'kidney collecting duct epithelial cell','kidney collecting duct principal cell',
        'kidney loop of Henle ascending limb epithelial cell','macrophage','mesangial cell'
    ],
    'MAT': [
        'B cell','CD4-positive, alpha-beta T cell','CD8-positive, alpha-beta T cell','NK cell','T cell',
        'endothelial cell','epithelial cell','macrophage','mesenchymal stem cell of adipose','myeloid cell'
    ],
    'Pancreas': [
        'endothelial cell','leukocyte','pancreatic A cell','pancreatic B cell','pancreatic D cell','pancreatic PP cell',
        'pancreatic acinar cell','pancreatic ductal cell','pancreatic stellate cell'
    ],
    'Large_Intestine': [
        'Brush cell of epithelium proper of large intestine','enterocyte of epithelium of large intestine',
        'enteroendocrine cell','epithelial cell of large intestine','intestinal crypt stem cell',
        'large intestine goblet cell','secretory cell'
    ],
    'Mammary_Gland': [
        'basal cell','endothelial cell','luminal epithelial cell of mammary gland','stromal cell'
    ],
    'Limb_Muscle': [
        'B cell','T cell','endothelial cell','macrophage','mesenchymal stem cell','skeletal muscle satellite cell'
    ],
    'Spleen': [
        'B cell','CD4-positive, alpha-beta T cell','CD8-positive, alpha-beta T cell','NK cell','granulocyte',
        'proerythroblast'
    ],
    'Thymus': [
        'DN4 thymocyte','endothelial cell','epithelial cell of thymus','fibroblast','macrophage','thymocyte'
    ],
    'Liver': [
        'B cell','CD4-positive, alpha-beta T cell','Kupffer cell','NK cell','T cell',
        'endothelial cell of hepatic sinusoid','hepatocyte','mature NK T cell','myeloid leukocyte','neutrophil'
    ],
    'Aorta': [
        'aortic endothelial cell','fibroblast of cardiac tissue','fibrocyte','macrophage',
        'professional antigen presenting cell'
    ],
    'BAT': [
        'B cell','NK cell','T cell','endothelial cell','epithelial cell','mesenchymal stem cell of adipose','myeloid cell'
    ]
}

MOUSE_CORTEX_BICCN_CELL_TYPES = 'L2slash3_IT', 'L6b', 'L5_IT', 'Pvalb', 'Sst', 'Vip', 'Lamp5', 'L6_IT', 'L6_CT', 'L5slash6_NP', 'Sncg'

TABULA_MURIS_TISSUE_ORGAN_SYSTEM_DICT = {
    "Aorta": "Cardiorespiratory",
    "BAT": "Adipose",
    "Bladder": "Immune & hematopoietic",
    "Brain_Myeloid": "Brain",
    "Brain_Non-Myeloid": "Brain",
    "Diaphragm": "Musculoskeletal",
    "GAT": "Adipose",
    "Heart": "Cardiorespiratory",
    "Kidney": "Excretory & integumentary",
    "Large_Intestine": "Digestive & metabolic",
    "Limb_Muscle": "Musculoskeletal",
    "Liver": "Digestive & metabolic",
    "Lung": "Cardiorespiratory",
    "MAT": "Adipose",
    "Mammary_Gland": "Excretory & integumentary",
    "Marrow": "Immune & hematopoietic",
    "Pancreas": "Digestive & metabolic",
    "SCAT": "Adipose",
    "Skin": "Excretory & integumentary",
    "Spleen": "Immune & hematopoietic",
    "Thymus": "Immune & hematopoietic",
    "Tongue": "Musculoskeletal",
    "Trachea": "Cardiorespiratory"
}

TABULA_MURIS_CELL_TYPE_ABBREVIATION_DICT = {
    "adventitial cell": "adventitial cell",
    "atrial myocyte": "AM",
    "aortic endothelial cell": "EC (aortic)",
    "astrocyte": "astrocyte",
    "basal cell": "basal cell",
    "basal cell of epidermis": "basal cell (epidermis)",
    "Bergmann glial cell": "BG",
    "B cell": "B cell",
    "bladder cell": "bladder cell",
    "bladder urothelial cell": "bladder urothelial cell",
    "brain pericyte": "brain pericyte",
    "bronchial smooth muscle cell": "SMC (bronchial)",
    "bulge keratinocyte": "bulge keratinocyte",
    "chondrocyte": "chondrocyte",
    "CD4-positive, alpha-beta T cell": "T cell (CD4+)",
    "CD8-positive, alpha-beta T cell": "T cell (CD8+)",
    "ciliated columnar cell of tracheobronchial tree": "CCC",
    "classical monocyte": "Mono (CL)",
    'DN4 thymocyte': "DN4 thymocyte",
    "dendritic cell": "DC",
    "endocardial cell": "Endoc",
    "endothelial cell": "EC",
    "endothelial cell of coronary artery": "EC (coronary)",
    "endothelial cell of hepatic sinusoid": "EC (hepatic sinusoid)",
    "endothelial cell of lymphatic vessel": "EC (lymphatic)",
    "enterocyte of epithelium of large intestine": "Enterocyte",
    "early pro-B cell": "B cell (early pro)",
    "epidermal cell": "epidermal cell",
    "epithelial cell": "Ep",
    "fibroblast": "Fb",
    "fibroblast of cardiac tissue": "Fb (cardiac)",
    "fibroblast of lung": "Fb (lung)",
    "granulocyte": "granulocyte",
    "granulocyte monocyte progenitor cell": "GMP",
    "granulocytopoietic cell": "granulocytopoietic cell",
    "hematopoietic stem cell": "HSC",
    "hepatocyte": "hepatocyte",
    "immature B cell": "B cell (immature)",
    "intestinal crypt stem cell": "intestinal crypt stem cell",
    "intermediate monocyte": "Mono (inter)",
    "interneuron": "neuron (inter)",
    "keratinocyte": "keratinocyte",
    "large intestine goblet cell": "Goblet cell",
    "late pro-B cell": "B cell (late pro)",
    "luminal epithelial cell of mammary gland": "Ep (luminal)",
    "macrophage": "macrophage",
    "mature alpha-beta T cell": "T cell (mature)",
    "mature NK T cell": "T cell (NK mature)",
    "medium spiny neuron": "neuron (medium spiny)",
    "megakaryocyte-erythroid progenitor cell": "MEP",
    "mesenchymal stem cell": "MSC",
    "mesenchymal stem cell of adipose": "MSC (adipose)",
    "microglial cell": "microglial cell",
    "monocyte": "Mono",
    "myeloid cell": "myeloid cell",
    "myeloid dendritic cell": "DC (myeloid)",
    "naive B cell": "B cell (naive)",
    "neuronal stem cell": "NSC",
    "NK cell": "NK cell",
    "non-classical monocyte": "Mono (NC)",
    "oligodendrocyte": "OC",
    "oligodendrocyte precursor cell": "OPC",
    "pancreatic A cell": "pancreatic A cell",
    "pancreatic B cell": "pancreatic B cell",
    "pancreatic D cell": "pancreatic D cell",
    "pancreatic ductal cell": "pancreatic ductal cell",
    "precursor B cell": "B cell (pre)",
    "promonocyte": "Mono (pro)",
    "pulmonary interstitial fibroblast": "Fb (pulmonary)",
    "regulatory T cell": "T cell (regulatory)",
    "secretory cell": "secretory cell",
    "skeletal muscle satellite cell": "SC (muscle)",
    "smooth muscle cell": "SMC",
    "smooth muscle cell of the pulmonary artery": "SMC (pulmonary)",
    "stromal cell": "stromal cell",
    "T cell": "T cell",
    "thymocyte": "thymocyte",
    "type I pneumocyte": "AT1",
    "type II pneumocyte": "AT2",
    "valve cell": "VIC", # DOUBLE CHECK IF VALVE INTERSTITIAL CELL
    "vein endothelial cell": "EC (vein)"
}

TABULA_MURIS_CELL_TYPE_LINEAGE_DICT = {
    "hematopoietic stem cell": "Hematopoietic Stem",
    "late pro-B cell": "B Cell",
    "precursor B cell": "B Cell",
    "immature B cell": "B Cell",
    "naive B cell": "B Cell",
    "B cell": "B Cell",
    "DN4 thymocyte": "T / NK Cell",
    "thymocyte": "T / NK Cell",
    "T cell": "T / NK Cell",
    "CD4-positive, alpha-beta T cell": "T / NK Cell",
    "CD8-positive, alpha-beta T cell": "T / NK Cell",
    "NK cell": "T / NK Cell",
    "myeloid cell": "Myeloid",
    "granulocytopoietic cell": "Myeloid",
    "granulocyte": "Myeloid",
    "promonocyte": "Myeloid",
    "monocyte": "Myeloid",
    "macrophage": "Myeloid",
    "microglial cell": "Myeloid",
    "mesenchymal stem cell": "Mesenchymal & Stromal",
    "mesenchymal stem cell of adipose": "Mesenchymal & Stromal",
    "stromal cell": "Mesenchymal & Stromal",
    "adventitial cell": "Mesenchymal & Stromal",
    "brain pericyte": "Mesenchymal & Stromal",
    "fibroblast": "Mesenchymal & Stromal",
    "fibroblast of cardiac tissue": "Mesenchymal & Stromal",
    "chondrocyte": "Mesenchymal & Stromal",
    "valve cell": "Mesenchymal & Stromal",
    "skeletal muscle satellite cell": "Mesenchymal & Stromal",
    "bronchial smooth muscle cell": "Mesenchymal & Stromal",
    "endothelial cell": "Endothelial",
    "aortic endothelial cell": "Endothelial",
    "endothelial cell of coronary artery": "Endothelial",
    "endothelial cell of hepatic sinusoid": "Endothelial",
    "endocardial cell": "Endothelial",
    "epidermal cell": "Epithelial & Epidermal",
    "basal cell": "Epithelial & Epidermal",
    "basal cell of epidermis": "Epithelial & Epidermal",
    "keratinocyte": "Epithelial & Epidermal",
    "bulge keratinocyte": "Epithelial & Epidermal",
    "luminal epithelial cell of mammary gland": "Epithelial & Epidermal",
    "secretory cell": "Epithelial & Epidermal",
    "intestinal crypt stem cell": "GI, Pancreatic & Urothelial",
    "enterocyte of epithelium of large intestine": "GI, Pancreatic & Urothelial",
    "large intestine goblet cell": "GI, Pancreatic & Urothelial",
    "hepatocyte": "GI, Pancreatic & Urothelial",
    "pancreatic ductal cell": "GI, Pancreatic & Urothelial",
    "pancreatic A cell": "GI, Pancreatic & Urothelial",
    "pancreatic B cell": "GI, Pancreatic & Urothelial",
    "pancreatic D cell": "GI, Pancreatic & Urothelial",
    "bladder cell": "GI, Pancreatic & Urothelial",
    "bladder urothelial cell": "GI, Pancreatic & Urothelial",
    "astrocyte": "Neural & Glial",
    "oligodendrocyte precursor cell": "Neural & Glial",
    "oligodendrocyte": "Neural & Glial",
}