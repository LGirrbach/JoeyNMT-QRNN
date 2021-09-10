mkdir -p data
cd data

# # Movie review data
# wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# tar -zxf aclImdb_v1.tar.gz
# rm aclImdb_v1.tar.gz

# Penn Treebank data
# wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
# tar -xzf simple-examples.tgz
# mkdir -p PTB
# mv simple-examples/data/* PTB/
# rm -r simple-examples
# rm simple-examples.tgz


# GloVe vectors
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
python -m gensim.scripts.glove2word2vec --input  glove.840B.300d.txt --output glove.840B.300d.w2vformat.txt
rm glove.840B.300d.txt
rm glove.840B.300d.zip


