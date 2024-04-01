# DeepEpiIL13: Deep Learning for Rapid and Accurate Prediction of IL-13 Inducing Epitopes using Pre-trained Language Models and Multi-Window Convolutional Neural Networks

## Introduction <a name="abstract"></a>
This study introduces a powerful deep learning framework for accurate epitope prediction, offering new avenues for the development of epitope-based vaccines and immunotherapies targeting IL-13-mediated disorders. The successful identification of IL-13-inducing epitopes paves the way for novel therapeutic interventions against allergic diseases, inflammatory conditions, and potentially severe viral infections like COVID-19. 
<br>
<br>
![workflow](Figure/Workflow.jpg)
<br>
The workflow for IL-13 inducing prediction model.

## Dataset <a name="Dataset"></a>

| Dataset        | Total       | IL-13-inducing peptides    | Non-IL-13-inducing peptides |
|----------------|-------------|----------------------------|-----------------------------|
| Training data  | 2576          | 250                      | 2326                        |
| Testing data   | 645           | 63                       | 582                         |
| SARS-CoV2 data | 52            | 12                       | 40                          |

## Quick start <a name="quickstart"></a>
### Step 1: Generate Data Features
Example usage:
```bash
python get_Binary_Matrix.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_mmseqs2.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_ProtTrans.py -in "Your FASTA file folder" -out "The destination folder of your output"
```
"Note: Ensure to update the path to your protein sequence database within get_mmseqs2.py as necessary."

### Step 2: Change Length

Example usage:
```bash
python length_change.py -in "Your FASTA file folder" -out "The destination folder of your output" -dim "The dimension of your input"
```
"Note: The dimesion of Binary_Matrix/Mmseqs2 are 20, ProtTrans is 1024"

### Step 3: Execute Prediction

#### Training MCNN Model

Example usage:
```bash
python DeepEpiIL13.py -D_tr "Train_Data folder path" -L_tr "Train_Label folder path " -D_ts "Test_Data folder path" -L_ts "Test_Label folder path " -n_feat "Dimensions"
```

"Note:  
-n_fil #the number of filters in the convolutional layer  
-n_hid #the number of hidden units in the dense layer  
-bs #the batch size  
-ws #the window sizes for convolutional filters  
-n_feat #the number of features  
-e #the number of epochs for training  
-val #the mod for validation 'cross' or 'independent'  
-k_fold #the number of k for k_fold cross validation"  

#### Run Our Model

Example usage:
```bash
python ./model/test_model.py -m_path ./model/set_model -x_ts_data "the Test_Data folder path" -y_ts_data "the Test_Label folder path"
```
