
# AST: Audio Spectrogram Transformer  
 - [Introduction](#Introduction)

## Introduction  

<p align="center"><img src="https://github.com/YuanGongND/ast/blob/master/ast.png?raw=true" alt="Illustration of AST." width="300"/></p>

## Citing  
The following paper proposes the Audio Spectrogram Transformer.
```  
@inproceedings{gong21b_interspeech,
  author={Yuan Gong and Yu-An Chung and James Glass},
  title={{AST: Audio Spectrogram Transformer}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={571--575},
  doi={10.21437/Interspeech.2021-698}
}
```  
  
## Getting Started  

Step 1. Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.

```
cd ast/ 
python3 -m venv venvast
source venvast/bin/activate
pip install -r requirements.txt 
```

Step 2. Download the ASVspoof2021 dataset into the `exp/asvspoof2021/data` directory.

```
./download_asvspoof.sh
```

 ## Acknowledgements
This is repository is a modified fork of the official AST repository `YuanGongND/ast`.

