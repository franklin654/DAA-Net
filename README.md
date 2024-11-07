# DAA-Net: Enhanced DuckNet with reverse Axial Attention

# Architecture

<div align=center><img src="https://github.com/franklin654/DAA-Net/blob/main/Images/Architecture.png" width="1000" height="450" alt="Result"/></div>

## Installation & Usage

## Datasets
The datasets used in this study are publicly available at: 

- WCE-Dataset: [here](https://zenodo.org/records/10156571). 
- Kvasir-SEG: [here](https://datasets.simula.no/kvasir-seg/). 
- CVC-ClinicDB: [here](https://polyp.grand-challenge.org/CVCClinicDB/). 
## Results

### WCE-Dataset

| Model                      | Miou  | Mdice | Accuracy | Precision |
|----------------------------|-------|-------|----------|-----------|
| **DAANet (16 filters)**    | 0.8245 | 0.8673 | 0.933    | 0.9137    |
| **DUCK-NET (17 filters)**  | 0.652  | 0.7932 | 0.901    | 0.803     |
| **CARANET**                | 0.659  | 0.748  | 0.862    | 0.926     |
| **FCB-FORMER**             | 0.67   | 0.695  | 0.756    | 0.444     |
| **SSFormer-S**             | 0.71   | 0.732  | 0.719    | 0.847     |
| **Resunet++**              | 0.368  | 0.521  | 0.82     | -         |
| **Unet**                   | 0.262  | 0.54   | -        | -         |

### Kvasir-SEG

| Model                      | Miou   | Mdice   |
|----------------------------|--------|---------|
| **DAANet (16 filters)**    | 0.8991 | 0.9401  |
| **FCB-FORMER**             | 0.8903 | 0.9385  |
| **DUCK-NET (17 filters)**  | 0.8769 | 0.9343  |
| **SSFormer**               | 0.8743 | 0.9261  |
| **CARANET**                | 0.865  | 0.918   |
| **Resunet++**              | 0.7927 | 0.8133  |
| **Unet**                   | 0.4334 | 0.7147  |

### CCVC-ClinicDB

| Model                      | Miou   | Mdice   |
|----------------------------|--------|---------|
| **DAANet (16 filters)**    | 0.910  | 0.940   |
| **FCB-FORMER** [7]         | 0.902  | 0.946   |
| **DUCK-NET (17 filters)**  | 0.8952 | 0.945   |
| **CARANET**                | 0.887  | 0.936   |
| **SSFormer-S**             | 0.8759 | 0.9268  |
| **Resunet++**              | 0.7962 | 0.7955  |
| **Unet**                   | 0.4711 | 0.6419  |

## Ablation Study

### WCE-Dataset
| Model Configuration               | Miou   | Mdice   |
|-----------------------------------|--------|---------|
| **Our Model**                     | 0.8245 | 0.8673  |
| **Our Model with CFP Module**     | 0.751  | 0.815   |
| **Our Model with Attention**      | 0.732  | 0.803   |

### Kvasir-SEG

| Model Configuration               | Miou   | Mdice   |
|-----------------------------------|--------|---------|
| **Our Model**                     | 0.8991 | 0.9401  |
| **Our Model with CFP Module**     | 0.868  | 0.924   |
| **Our Model with Attention**      | 0.773  | 0.842   |

### CCVC-ClinicDB

| Model Configuration               | Miou   | Mdice   |
|-----------------------------------|--------|---------|
| **Our Model**                     | 0.910  | 0.946   |
| **Our Model with CFP Module**     | 0.843  | 0.895   |
| **Our Model with Attention**      | 0.837  | 0.906   |

## Qualitative Results

### WCE-Dataset
<div align=center><img src="https://github.com/franklin654/DAA-Net/blob/main/Images/Qualitative_analysis_WCE.jpg" width="700" height="750" alt="Result"/></div>

### Kvasir-SEG

<div align=center><img src="https://github.com/franklin654/DAA-Net/blob/main/Images/Qualitative_analysis_kvasir.png" width="700" height="750" alt="Result"/></div>

## Activation Maps
<div align=center><img src="https://github.com/franklin654/DAA-Net/blob/main/Images/Activation_Maps.png" width="700" height="750" alt="Result"/></div>

## Model Weights

The weights for the models that achieved the final results can be found on here  on Google Drive.

## Citation



