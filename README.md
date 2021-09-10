# Seq2Seq Seminar Project
### This project contains
1. An updated version of the repositories [pytorch-qrnn](https://github.com/salesforce/pytorch-qrnn/) and [awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm) to PyTorch v1.4.0
2. Custom reimplementation of the QRNN model and experiments in from [Merity et al.](https://arxiv.org/abs/1611.01576) using [Allennlp](https://allennlp.org/)
3. Integration of the QRNN Encoder-Decoder variant into JoeyNMT ([Kreuzer et al.](https://github.com/joeynmt/joeynmt))
4. An attempt to implement the cold-fusion method proposed in [Sriram et al.](https://arxiv.org/abs/1708.06426)

Furthermore, all data needed for the different scripts are provided in this repository except for GloVe embeddings.

## Updated version of Salesforce repositories
### Requirements
* Python 3.7 (or compatible)
* PyTorch 1.4.0 (or compatible)
* CuPy 7.5.0 (or compatible)

CuPy doesn't seem to work without CUDA setup, so if you want to run the code in a CPU-only environment, you may want to not import CuPy.

### Usage
You can use the code as suggested in the original repository except the `generate.py`, `pointer.py` and `finetune.py` scripts are removed. See the README file in the `awd-lstm-lm` folder for details.

I have moved the QRNN scripts into the `awd-lstm-lm` folder, so there is no need to install any (other) package (in the original implementation, the QRNN and AWD-LSTM are different packages).

### Known problems
* Generally, the Salesforce implementation is not complete. Especially, the `generate.py`, `pointer.py` and `finetune.py` scripts are dysfunctional in the original repository (as stated in the README), so I removed them. <br> If you care about the details:
  1. `RNNModel` is programmed to return contextual embeddings (activations of the last hidden layer). This works with the `SplitCrossEntropy-loss` employed in `main.py`, because it has a separate linear layer to produce prediction scores for the vocabulary. Contrarily, the removed scripts expect the output of the model to be prediction scores for the vocabulary.
  2. As far as I can tell, `finetune.py` doesn't do anything different (except fot not working) than the main training procedure, so I don't see any reason to use it. Also, because of lacking documentation, I am not sure what `pointer.py` is meant to do.
* There are a few known minor bugs which I fixed also in this code
* The QRNN doesn't train with the proposed hyperparameters. Changing the learning rate helps with the problem, but the general results seem not to be much different from the results that I achieve with my custom implementation (ca. 110 PPL). This is far worse than reported in the paper and in the Salesforce repository README.

## Custom reimplementation of QRNN
### What is reused from the Salesforce Pytorch implementation
1. General structure of `QRNN` and `QRNNLayer`
2. Cuda-Kernel for `f-pooling`
3. Design principle: Can be used as a drop-in replacement for any PyTorch RNN variant

### What is new/improved compared to the Salesforce implementation (PyTorch)
1. New: Arbitrary convolutional kernel sizes (the Salesforce implementation only supports kernel widths 1 and 2)
2. New: Bidirectional mode
3. New: Cuda-Kernel for `ifo-pooling`
4. New: Implementation of the QRNN-Decoder variant (the Salesforce implementation can only be used for language modelling). This includes the attention scheme proposed in the paper.
5. Improved: Uses PyTorch's built-in 1d-Convolution (instead of a custom operation)
6. Improved: Doesn't have to explicitely reset the cached previous inputs (like in the Salesforce implementation)
7. New: Integration into Allennlp and JoeyNMT

### Custom QRNN reimplementation
#### Requirements
You need:
 * Python 3.7 (or compatible)
 * PyTorch 1.4.0 (or compatible)
 * CuPy 7.5.0 (or compatible)
 * Allennlp 0.9.0 (be sure not to use version 1.x, as this is incompatible)
 * gensim 3.8.0 (or compatible)
 * any requirements of the above mentioned libraries

You may want to download the GloVe vectors for review classification by running the `download_data.sh` script.

#### Usage
You can just run:
```
python review_classification.py
python language_modelling.py
```
Be sure that the local `saved_models/lm` and `saved_models/review_classification` directories exist and are empty. If they are not empty, `allennlp` will try to load checkpoint models from the directories and not train at all.

For changing hyperparameters, you have to edit the respective python scripts, but the parameters themselves should be self-explanatory.

### QRNN-Decoder implementation
#### Requirements
You need:
 * Python 3.7 (or compatible)
 * PyTorch 1.4.0 (or compatible)
 * CuPy 7.5.0 (or compatible)
 * the JoeyNMT version provided in this repository. You can install it just like the regular JoeyNMT package.
 * all requirements of JoeyNMT

#### Usage
You can just run the provided `yaml`-configuration file as one would normally run JoeyNMT, e.g.
```
python -u -m joeynmt train configs/qrnn.yaml
```
Generally, usage is the same as for the normal JoeyNMT distribution, but the configuration options are restricted to the use of QRNNs.

## Cold Fusion
This implementation is experimental, so it works, but does not yield any configuration options. Also, the translation model doesn't seem to train well, so there may be some shortcomings.

### Requirements
You need:
 * Python 3.7 (or compatible)
 * PyTorch 1.4.0 (or compatible)
 * CuPy 7.5.0 (or compatible)
 * Allennlp 0.9.0 (be sure not to use version 1.x, as this is incompatible)
 * gensim 3.8.0 (or compatible)

#### Usage
Just run `python cold_fusion.py`.
