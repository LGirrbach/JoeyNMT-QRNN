
"""
Based on
http://www.realworldnlpbook.com/blog/building-seq2seq-machine-translation-models-using-allennlp.html
https://github.com/mhagiwara/realworldnlp/blob/master/examples/mt/mt.py
"""

import itertools
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np

from custom_language_model import LanguageModel
from custom_pytorch_seq2seq_wrapper import CustomPytorchSeq2SeqWrapper

from overrides import overrides

from allennlp.training import Trainer
from allennlp.models.model import Model
from allennlp.training.util import evaluate
from allennlp.nn.activations import Activation
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask
from allennlp.data.iterators import BucketIterator
from allennlp.data.tokenizers import WordTokenizer
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SimpleSeq2SeqPredictor
from allennlp.modules.token_embedders import Embedding
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.modules.attention import DotProductAttention
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.dataset_readers import SimpleLanguageModelingDatasetReader
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import _PyTorchLearningRateSchedulerWrapper


MINIBATCHSIZE = 16
PATIENCE = 5
MAX_DECODING_STEPS = 450
SRC_EMBEDDING_DIM = 16
TRG_EMBEDDING_DIM = 16
HIDDEN_DIM = 320
GATE_DIM = 128

USE_GPU = torch.cuda.is_available()
assert USE_GPU


class GatedSeq2Seq(SimpleSeq2Seq):
    def __init__(self, *args, **kwargs):
        cold_fusion = kwargs.pop("cold_fusion")
        super(GatedSeq2Seq, self).__init__(*args, **kwargs)
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        self.cold_fusion = cold_fusion
        self._output_projection_layer = nn.Linear(self.cold_fusion.output_dim, num_classes)
    
    @overrides
    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        if self._attention:
            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)

            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input), -1)
        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = embedded_input

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input,
                (decoder_hidden, decoder_context))

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context
        
        # Cold Fusion gating mechanism
        lm_hidden = state.get("lm_hidden", None)
        lm_context = state.get("lm_context", None)
        (lm_hidden, lm_context), gate_hidden = self.cold_fusion(last_predictions.unsqueeze(1),
                                                                decoder_hidden,
                                                                lm_hidden, lm_context)
        state["lm_hidden"] = lm_hidden.transpose(0, 1).contiguous()
        state["lm_context"] = lm_context.transpose(0, 1).contiguous()

        # shape: (group_size, num_classes)
        # print(gate_hidden.shape)
        output_projections = self._output_projection_layer(gate_hidden)
        # print(output_projections.shape)

        return output_projections, state


class ColdFusion(nn.Module):
    def __init__(self, language_model, vocab_size, input_dim, gate_dim, hidden_dim):
        super(ColdFusion, self).__init__()
        
        language_model_output_dim = vocab_size
        self.language_model = language_model
        self.lm_logits_projection = nn.Linear(language_model_output_dim,
                                              gate_dim)
        self.gate_linear = nn.Linear(input_dim + gate_dim, gate_dim)
        self.dnn = nn.Linear(input_dim + gate_dim, hidden_dim)
        self.output_dim = hidden_dim
        
    
    def forward(self, last_predictions, decoder_hidden, lm_hidden, lm_context):
        if lm_hidden is None:
            hidden = None
        else:
            hidden = (lm_hidden.transpose(0, 1).contiguous(),
                      lm_context.transpose(0, 1).contiguous())
        with torch.no_grad():
            return_dict = \
                self.language_model({
                    'tokens': last_predictions,
                    'hidden': hidden
                    }, training=False)
            lm_logits = return_dict['forward_probs']
            lm_hidden = return_dict['hidden']
        assert lm_logits.shape[1] >= 1
        lm_projection = self.lm_logits_projection(lm_logits).squeeze(1)
        gate = torch.sigmoid(self.gate_linear(torch.cat([decoder_hidden, lm_projection], dim=1)))
        fused_state = torch.cat([decoder_hidden, gate*lm_projection], dim=1)
        fused_state = torch.relu(self.dnn(fused_state)).contiguous()
        return lm_hidden, fused_state


def make_translation_datasets():
    reader = Seq2SeqDatasetReader(
        source_tokenizer=CharacterTokenizer(),
        target_tokenizer=CharacterTokenizer(),
        source_token_indexers={'tokens': SingleIdTokenIndexer()},
        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})
    trn_dataset = reader.read('data/train_de_en.tsv')
    vld_dataset = reader.read('data/dev_de_en.tsv')
    tst_dataset = reader.read('data/test_de_en.tsv')
    return trn_dataset, vld_dataset, tst_dataset, reader


def make_language_modelling_datasets():
    reader = SimpleLanguageModelingDatasetReader(
        tokenizer=CharacterTokenizer(),
        token_indexers={"tokens": SingleIdTokenIndexer()}
        )
    reader.lazy = False
    
    train_dataset = reader.read("data/train.trg")
    valid_dataset = reader.read("data/dev.trg")
    
    return train_dataset, valid_dataset


def make_translation_model(vocab, language_model):
    src_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                              embedding_dim=SRC_EMBEDDING_DIM, padding_index=0)
    src_embedder = BasicTextFieldEmbedder({"tokens": src_embedding})
    encoder = PytorchSeq2SeqWrapper(
        nn.LSTM(SRC_EMBEDDING_DIM, HIDDEN_DIM, num_layers=4, batch_first=True)
        )
    cold_fusion = ColdFusion(language_model, vocab.get_vocab_size(), HIDDEN_DIM, GATE_DIM, HIDDEN_DIM)
    attention = DotProductAttention()
    model = GatedSeq2Seq(vocab, src_embedder, encoder, MAX_DECODING_STEPS,
                          target_embedding_dim=TRG_EMBEDDING_DIM,
                          target_namespace='target_tokens',
                          attention=attention,
                          beam_size=8,
                          cold_fusion=cold_fusion,
                          use_bleu=True)
    return model


def make_language_model(vocab):
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size(),
                                embedding_dim=TRG_EMBEDDING_DIM, padding_index=0)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    
    # qrnn = QRNN(input_size=word_embeddings.get_output_dim(), pooling='ifo',
    #            hidden_size=HIDDENSIZE, num_layers=4, dense=False, batch_first=True, dropout=0.4, zoneout=0.3)
    encoder = CustomPytorchSeq2SeqWrapper(
        nn.LSTM(TRG_EMBEDDING_DIM, HIDDEN_DIM, num_layers=4, batch_first=True)
        )

    model = LanguageModel(
        vocab=vocab,
        text_field_embedder=word_embeddings,
        contextualizer=encoder
        )
    return model


def train_translation_model(model, vocab, train_dataset, validation_dataset, reader,
                            epochs=20):
    if USE_GPU:
        model.cuda()
        cuda_device=0
    else:
        cuda_device = -1
        
    optimizer = optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=MINIBATCHSIZE, sorting_keys=[("source_tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        patience=PATIENCE,
        shuffle=True,
        serialization_dir="saved_models/translate/",
        # grad_norm=10,
        # learning_rate_scheduler=scheduler,
        cuda_device= cuda_device,
        num_epochs=epochs
        )
    trainer.train()

    #for i in range(epochs):
        #print('Epoch: {}'.format(i))
        #trainer.train()

        #predictor = SimpleSeq2SeqPredictor(model, reader)

        #for instance in itertools.islice(validation_dataset, 10):
            #print('SOURCE:', ''.join([str(token) for token in instance.fields['source_tokens'].tokens]))
            #print('GOLD:', ''.join([str(token) for token in instance.fields['target_tokens'].tokens]))
            #print('PRED:', ''.join([str(token) for token in predictor.predict_instance(instance)['predicted_tokens']]))


def train_language_model(model, vocab, train_dataset, valid_dataset):
    if USE_GPU:
        model.cuda()
        cuda_device=0
    else:
        cuda_device = -1
        
    iterator = BucketIterator(batch_size=MINIBATCHSIZE,
                              sorting_keys=[("source", "num_tokens")])
    iterator.index_with(vocab)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=2*(10**(-4)))
    # optimizer = optim.SGD(model.parameters(), lr=1, weight_decay=2*(10**(-4)))
    # optimizer = optim.Adam(model.parameters())
    scheduler = MultiStepLR(optimizer, milestones=np.arange(6, 550), gamma=0.95)
    scheduler = _PyTorchLearningRateSchedulerWrapper(scheduler)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=valid_dataset,
        patience=PATIENCE,
        shuffle=True,
        serialization_dir="saved_models/lm",
        grad_norm=10,
        # learning_rate_scheduler=scheduler,
        cuda_device= cuda_device,
        num_epochs=50
        )
    
    trainer.train()

if __name__ == '__main__':
    TRAIN_LM = True
    trn_dataset, vld_dataset, tst_dataset, reader = make_translation_datasets()
    vocab = Vocabulary.from_instances(trn_dataset + vld_dataset,
                                      min_count={'tokens': 50, 'target_tokens': 50})
    if TRAIN_LM:
        lm_trn_dataset, lm_vld_dataset = make_language_modelling_datasets()
        lm = make_language_model(vocab)
        print("\n\nTraining language model")
        train_language_model(lm, vocab, lm_trn_dataset, lm_vld_dataset)
    
    print("\n\nTraining translation model")
    lm = make_language_model(vocab)
    lm.load_state_dict(torch.load("saved_models/lm/best.th"))
    model = make_translation_model(vocab, lm)
    train_translation_model(model, vocab, trn_dataset, vld_dataset, reader)
    
    
