# Transformers-Encoder Layer and Decoder Layer


## Project Overview

This project provides an implementation of an Encoder layers and Decoder layers of a Transformer. It includes detailed implementations of both the encoder and decoder components, utilizing multi-head attention mechanisms. This model structure can be applied to various sequence-to-sequence tasks, such as language translation.

graph TD
    A[Input] --> B[TransformerEncoder]
    B --> C{{"For each layer<br>1 to n_layers"}}
    C --> |Layer i| D[TransformerEncoderLayer]
    D --> E[QKV Linear]
    E --> F[Reshape & Permute]
    F --> G[MultiHeadSelfAttention]
    G --> H[Reshape]
    H --> I[Dropout]
    I --> J{{"Add & Norm"}}
    J --> K[Feed Forward]
    K --> L[Dropout]
    L --> M{{"Add & Norm"}}
    M --> |Next Layer| C
    C --> |Output| N[Final Output]

    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef io fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    class A,N io;
    class B,D,E,F,G,H,I,K,L process;
    class C,J,M decision;

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

