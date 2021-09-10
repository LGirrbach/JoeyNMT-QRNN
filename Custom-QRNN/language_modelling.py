import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np
from qrnn import QRNN

from allennlp.training import Trainer
from allennlp.models import LanguageModel
from allennlp.training.util import evaluate
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.iterators import BucketIterator
from allennlp.modules.token_embedders import Embedding
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.dataset_readers import SimpleLanguageModelingDatasetReader
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import _PyTorchLearningRateSchedulerWrapper


MINIBATCHSIZE = 20
HIDDENSIZE = 1240
EMBEDDINGDIM = 400
PATIENCE = 10
USE_GPU = torch.cuda.is_available()
# assert USE_GPU


def make_datasets():
    token_indexer = SingleIdTokenIndexer()
    word_splitter = JustSpacesWordSplitter()
    tokenizer = WordTokenizer(word_splitter=word_splitter, start_tokens=[START_SYMBOL],
                              end_tokens=[END_SYMBOL])
    
    reader = SimpleLanguageModelingDatasetReader(
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer}
        )
    reader.lazy = False
    
    train_dataset = reader.read("data/PTB/ptb.train.txt")
    valid_dataset = reader.read("data/PTB/ptb.valid.txt")
    test_dataset = reader.read("data/PTB/ptb.test.txt")
    
    return reader, train_dataset, valid_dataset, test_dataset


if __name__ == '__main__':
    # assert USE_GPU
    reader, train_dataset, valid_dataset, test_dataset = make_datasets()
    vocab = Vocabulary.from_instances(train_dataset + valid_dataset)
    #vocab.extend_from_instances(valid_dataset)
    #vocab.extend_from_instances(test_dataset)
    
    iterator = BucketIterator(batch_size=MINIBATCHSIZE,
                              sorting_keys=[("source", "num_tokens")])
    iterator.index_with(vocab)
    
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size(),
                                embedding_dim=EMBEDDINGDIM, padding_index=0)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    
    qrnn = QRNN(input_size=word_embeddings.get_output_dim(), pooling='ifo',
                hidden_size=HIDDENSIZE, num_layers=4, dense=True, batch_first=True, dropout=0.4, zoneout=0.3)
    encoder = PytorchSeq2SeqWrapper(qrnn)

    model = LanguageModel(
        vocab=vocab,
        text_field_embedder=word_embeddings,
        contextualizer=encoder
        )
    
    if USE_GPU:
        model.cuda()
        cuda_device=0
    else:
        cuda_device = -1
    
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=2*(10**(-4)))
    # optimizer = optim.SGD(model.parameters(), lr=1.0, weight_decay=2*(10**(-4)))
    optimizer = optim.Adam(model.parameters())
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
        num_epochs=550
        )
    
    trainer.train()
    
    # Evaluation
    print()
    print("=================================================================")
    print("Test metrics:")
    evaluation = evaluate(model, test_dataset, iterator, cuda_device, "")
    print(evaluation)
