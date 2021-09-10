import torch
import torch.nn as nn
import torch.optim as optim

import os
import gensim

from qrnn import QRNN
from overrides import overrides
from gensim.models import KeyedVectors

from allennlp.models import Model
from allennlp.data import Instance
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Token
from allennlp.training.util import evaluate
from allennlp.data.fields import LabelField
from allennlp.models import BasicClassifier
from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import TokenIndexer
from allennlp.modules.token_embedders import Embedding
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


EMBEDDINGDIM = 300
HIDDENSIZE = 256
MINIBATCHSIZE = 24
NUM_LAYERS = 4
USE_GPU = torch.cuda.is_available()

GLOVEFILE = "data/glove.840B.300d.w2vformat.txt"


class ReviewDatasetReader(DatasetReader):
    def __init__(self, tokenizer=lambda x: x.split(), token_indexers=None):
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(self, tokens, label):
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}
        
        if label is None:
            label = 0
        label_field = LabelField(label=label, skip_indexing=True)
        fields["label"] = label_field

        return Instance(fields)
    
    @overrides
    def _read(self, path):
        for sentiment in ["pos", "neg"]:
            label = 1 if sentiment == "pos" else 0
            for filename in os.listdir(os.path.join(path, sentiment)):
                with open(os.path.join(path, sentiment, filename)) as reviews:
                    review = reviews.read().strip()
                    yield self.text_to_instance(
                        [Token(x) for x in self.tokenizer(review.strip())],
                        label
                    )


def make_word_embeddings(vocab, use_glove=True):
    if use_glove:
        glove = KeyedVectors.load_word2vec_format(GLOVEFILE)
        glove_embeddings = torch.zeros(vocab.get_vocab_size(), EMBEDDINGDIM)
        not_found_indices = []
        for index, token in vocab.get_index_to_token_vocabulary().items():
            if index == 0 or token == '@@UNKNOWN@@' or token not in glove:
                not_found_indices.append(index)
            else:
                glove_embeddings[index] = torch.from_numpy(glove[token])

        mean_embedding = glove_embeddings.mean(dim=0)
        not_found_indices = torch.LongTensor(not_found_indices)
        glove_embeddings[not_found_indices] = mean_embedding
        glove_embeddings.requires_grad = True

        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size(),
                                    embedding_dim=EMBEDDINGDIM,
                                    padding_index=0,
                                    weight=glove_embeddings)
    else:
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size(),
                                    embedding_dim=EMBEDDINGDIM,
                                    padding_index=0)
        
    
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    return word_embeddings


def make_datasets():
    token_indexer = SingleIdTokenIndexer()
    word_splitter = SpacyWordSplitter(language='en_core_web_sm',
                                      pos_tags=False)
    word_splitter = word_splitter.split_words
    tokenizer = lambda x: [w.text for w in word_splitter(x)]
    
    reader = ReviewDatasetReader(
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer}
        )
    
    train_dataset = reader.read("data/aclImdb/train")
    test_dataset = reader.read("data/aclImdb/test")
    
    return reader, train_dataset, test_dataset


def make_encoder(qrnn=True):
    if qrnn:
        encoder = QRNN(EMBEDDINGDIM, HIDDENSIZE, num_layers=NUM_LAYERS, bidirectional=False,
                       dense=True, batch_first=True, dropout=0.3, first_kernel_size=2)
    else:
        encoder = nn.LSTM(EMBEDDINGDIM, HIDDENSIZE, num_layers=NUM_LAYERS,
                       bidirectional=True, batch_first=True)
    return PytorchSeq2VecWrapper(encoder)


if __name__ == '__main__':
    # assert USE_GPU
    reader, train_dataset, test_dataset = make_datasets()
    vocab = Vocabulary.from_instances(train_dataset)

    iterator = BucketIterator(batch_size=MINIBATCHSIZE,
                              sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)
    
    encoder = make_encoder(qrnn=True)
    word_embeddings = make_word_embeddings(vocab, use_glove=False)
    model = BasicClassifier(
        vocab=vocab,
        text_field_embedder=word_embeddings,
        seq2vec_encoder=encoder,
        num_labels=2
        )

    if USE_GPU:
        model.cuda()
        cuda_device = 0
    else:
        cuda_device = -1

    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9,
                              eps=1e-08, weight_decay=4*(10**(-6)))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        shuffle=True,
        patience=5,
        num_epochs=100,
        serialization_dir="saved_models/review_classification/",
        cuda_device=cuda_device
        )
    
    trainer.train()
    
    # Evaluation
    print()
    print("=================================================================")
    print("Test metrics:")
    evaluation = evaluate(model, test_dataset, iterator, cuda_device, "")
    print(evaluation)
