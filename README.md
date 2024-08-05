# Transformers-Encoder Layer and Decoder Layer


## Project Overview

This project provides an implementation of an Encoder layers and Decoder layers of a Transformer. It includes detailed implementations of both the encoder and decoder components, utilizing multi-head attention mechanisms. This model structure can be applied to various sequence-to-sequence tasks, such as language translation.

![image](https://github.com/user-attachments/assets/db714da4-2c53-42e8-b778-0b462a8df7e7) ![image](https://github.com/user-attachments/assets/619e2cc2-939b-40bc-93c0-b4e49f74b3c2)



## Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.7+
- Required Python packages (listed in `requirements.txt`)

## Installation

1. **Clone the repository:**

   ```bash
   git clone Prathmesh-Kolekar/Transformers-Encoder-Decoder
   cd Transformers-Encoder-Decoder
   ```

2. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the Encoder-Decoder model, you can run the provided Jupyter notebooks and Python scripts. Below are brief descriptions of each file:

### File Descriptions

- **`Encoder_arch_8heads.ipynb`:** This notebook contains the implementation of the encoder, utilizing 8 attention heads.

- **`Decoder_arch_8heads.ipynb`:** This notebook includes the implementation of the decoder architecture, also with 8 attention heads.

- **`feedforward.py`:** This script implements the feedforward neural network layer used in the model.
 
- **`Multi_head_crossattn.py`:** This script implements the Multi Head Cross Attention used in the model.

- **`Multi_head_selfattn.py`:** This script implements the Multi Head Self Attention used in the model.
 

### Running the Model

To train or evaluate the model, follow the instructions in the respective notebooks or scripts. For example, you can start by exploring the encoder architecture in `Encoder_arch_8heads.ipynb`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

