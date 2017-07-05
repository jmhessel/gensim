#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Paragarph vector with meta-features
"""

import logging
import os
import warnings

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from collections import namedtuple, defaultdict
from timeit import default_timer

from numpy import zeros, sum as np_sum, add as np_add, concatenate, \
    repeat as np_repeat, array, float32 as REAL, empty, ones, memmap as np_memmap, \
    sqrt, newaxis, ndarray, dot, vstack, dtype, divide as np_divide, integer


from gensim.utils import call_on_class_only
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.models.word2vec import Word2Vec, train_cbow_pair, train_sg_pair, train_batch_sg
from gensim.models.doc2vec import Doc2Vec, DocvecsArray, Doctag
from gensim.models.keyedvectors import KeyedVectors
from six.moves import xrange, zip
from six import string_types, integer_types

logger = logging.getLogger(__name__)

try:
    from gensim.models.meta_doc2vec_inner import train_meta_document_dm_concat, train_meta_document_dm, train_meta_document_dbow, get_max_feature_size
    #from gensim.models.doc2vec_inner import train_document_dbow, train_document_dm, train_document_dm_concat
    #from gensim.models.word2vec_inner import FAST_VERSION  # blas-adaptation shared from word2vec
    logger.debug('Fast version of {0} is being used'.format(__name__))
except ImportError:
    logger.warning('Slow version of {0} is being used'.format(__name__))
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1

    def get_max_feature_size():
        return None

    def train_meta_document_dbow(model, doc_words, doctag_indexes, doc_features, alpha, work=None,
                                train_words=False, learn_doctags=True, learn_words=True, learn_hidden=True,
                                word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """
        Update distributed bag of words model ("PV-DBOW") by training on a single document.

        Called internally from `Doc2Vec.train()` and `Doc2Vec.infer_vector()`.

        The document is provided as `doc_words`, a list of word tokens which are looked up
        in the model's vocab dictionary, and `doctag_indexes`, which provide indexes
        into the doctag_vectors array.

        If `train_words` is True, simultaneously train word-to-word (not just doc-to-word)
        examples, exactly as per Word2Vec skip-gram training. (Without this option,
        word vectors are neither consulted nor updated during DBOW doc vector training.)

        Any of `learn_doctags', `learn_words`, and `learn_hidden` may be set False to
        prevent learning-updates to those respective model weights, as if using the
        (partially-)frozen model to infer other compatible vectors.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from doc2vec_inner instead.

        """
        raise NotImplementedError("DBOW not implemented for meta info just yet... what does it mean?")
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf
        if train_words and learn_words:
            train_batch_sg(model, [doc_words], alpha, work)
        for doctag_index in doctag_indexes:
            for word in doc_words:
                train_sg_pair(model, word, doctag_index, alpha, learn_vectors=learn_doctags,
                              learn_hidden=learn_hidden, context_vectors=doctag_vectors,
                              context_locks=doctag_locks)

        return len(doc_words)

    def train_meta_document_dm(model, doc_words, doctag_indexes, doc_features, alpha, work=None, neu1=None,
                               learn_doctags=True, learn_words=True, learn_hidden=True,
                               word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """
        Update distributed memory model ("PV-DM") by training on a single document.

        Called internally from `Doc2Vec.train()` and `Doc2Vec.infer_vector()`. This
        method implements the DM model with a projection (input) layer that is
        either the sum or mean of the context vectors, depending on the model's
        `dm_mean` configuration field.  See `train_document_dm_concat()` for the DM
        model with a concatenated input layer.

        The document is provided as `doc_words`, a list of word tokens which are looked up
        in the model's vocab dictionary, and `doctag_indexes`, which provide indexes
        into the doctag_vectors array.

        Any of `learn_doctags', `learn_words`, and `learn_hidden` may be set False to
        prevent learning-updates to those respective model weights, as if using the
        (partially-)frozen model to infer other compatible vectors.

        This is the non-optimized, Python version. If you have a C compiler, gensim
        will use the optimized version from doc2vec_inner instead.

        """
        if word_vectors is None:
            word_vectors = model.wv.syn0
        if word_locks is None:
            word_locks = model.syn0_lockf
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf

        word_vocabs = [model.wv.vocab[w] for w in doc_words if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2**32]

        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original doc2vec code
            start = max(0, pos - model.window + reduced_window)
            end = (pos + model.window + 1 - reduced_window) if not model.asymmetric_window else pos
            window_pos = enumerate(word_vocabs[start:end], start)
            word2_indexes = [word2.index for pos2, word2 in window_pos if pos2 != pos]
            l1 = np_sum(word_vectors[word2_indexes], axis=0) + np_sum(doctag_vectors[doctag_indexes], axis=0) + doc_features
            count = len(word2_indexes) + len(doctag_indexes) + 1.
            if model.cbow_mean and count > 1 :
                l1 /= count
            neu1e = train_cbow_pair(model, word, word2_indexes, l1, alpha,
                                    learn_vectors=False, learn_hidden=learn_hidden)
            if not model.cbow_mean and count > 1:
                neu1e /= count
            if learn_doctags:
                for i in doctag_indexes:
                    doctag_vectors[i] += neu1e * doctag_locks[i]
            if learn_words:
                for i in word2_indexes:
                    word_vectors[i] += neu1e * word_locks[i]

        return len(word_vocabs)

    def train_meta_document_dm_concat(model, doc_words, doctag_indexes, doc_features, alpha, work=None, neu1=None,
                                      learn_doctags=True, learn_words=True, learn_hidden=True,
                                      word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """
        Update distributed memory model ("PV-DM") by training on a single document, using a
        concatenation of the context window word vectors (rather than a sum or average).

        Called internally from `Doc2Vec.train()` and `Doc2Vec.infer_vector()`.

        The document is provided as `doc_words`, a list of word tokens which are looked up
        in the model's vocab dictionary, and `doctag_indexes`, which provide indexes
        into the doctag_vectors array.

        Any of `learn_doctags', `learn_words`, and `learn_hidden` may be set False to
        prevent learning-updates to those respective model weights, as if using the
        (partially-)frozen model to infer other compatible vectors.

        This is the non-optimized, Python version. If you have a C compiler, gensim
        will use the optimized version from doc2vec_inner instead.

        """

        if word_vectors is None:
            word_vectors = model.wv.syn0
        if word_locks is None:
            word_locks = model.syn0_lockf
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf

        word_vocabs = [model.wv.vocab[w] for w in doc_words if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2**32]
        doctag_len = len(doctag_indexes)
        if doctag_len != model.dm_tag_count:
            return 0  # skip doc without expected number of doctag(s) (TODO: warn/pad?)

        null_word = model.wv.vocab['\0']
        pre_pad_count = model.window
        post_pad_count = model.window
        padded_document_indexes = (
            (pre_pad_count * [null_word.index])  # pre-padding
            + [word.index for word in word_vocabs if word is not None]  # elide out-of-Vocabulary words
            + (post_pad_count * [null_word.index])  # post-padding
        )

        for pos in range(pre_pad_count, len(padded_document_indexes) - post_pad_count):
            if model.asymmetric_window:
                word_context_indexes = (
                    padded_document_indexes[(pos - pre_pad_count): pos]  # preceding words
                )
            else:
                word_context_indexes = (
                    padded_document_indexes[(pos - pre_pad_count): pos]  # preceding words
                    + padded_document_indexes[(pos + 1):(pos + 1 + post_pad_count)]  # following words
                )

            word_context_len = len(word_context_indexes)
            predict_word = model.wv.vocab[model.wv.index2word[padded_document_indexes[pos]]]
            # numpy advanced-indexing copies; concatenate, flatten to 1d
            l1 = concatenate((doctag_vectors[doctag_indexes], word_vectors[word_context_indexes])).ravel()
            l1 = concatenate((l1, doc_features))
            neu1e = train_cbow_pair(model, predict_word, None, l1, alpha,
                                    learn_hidden=learn_hidden, learn_vectors=False)
            # discard the errors from the features...
            neu1e = neu1e[:-model.meta_size]

            # filter by locks and shape for addition to source vectors
            e_locks = concatenate((doctag_locks[doctag_indexes], word_locks[word_context_indexes]))
            neu1e_r = (neu1e.reshape(-1, model.vector_size)
                       * np_repeat(e_locks, model.vector_size).reshape(-1, model.vector_size))

            if learn_doctags:
                np_add.at(doctag_vectors, doctag_indexes, neu1e_r[:doctag_len])
            if learn_words:
                np_add.at(word_vectors, word_context_indexes, neu1e_r[doctag_len:])

        return len(padded_document_indexes) - pre_pad_count - post_pad_count


class TaggedMetaDocument(namedtuple('TaggedMetaDocument', 'words tags features')):
    """
    A single document, made up of `words` (a list of unicode string tokens)
    `tags` (a list of tokens), and `features` (a numpy array of document-level meta-features).
    Tags may be one or more unicode string
    tokens, but typical practice (which will also be most memory-efficient) is
    for the tags list to include a unique integer id as the only tag.

    Replaces "sentence as a list of words" from Word2Vec.

    """
    def __str__(self):
        return '%s(%s, %s, %s)' % (self.__class__.__name__, self.words, self.tags, self.features)


class MetaDoc2Vec(Doc2Vec):
    """Class for training, using and evaluating neural networks described in http://arxiv.org/pdf/1405.4053v2.pdf"""
    def __init__(self,
                 size,
                 documents=None,
                 trim_rule=None,
                 dm_concat=1,
                 meta_size=None, **kwargs):
        """
        Initialize the model from an iterable of `documents`.
        meta_size gives the size of the meta features array per document.
        """

        super(MetaDoc2Vec, self).__init__(size=size, trim_rule=trim_rule, dm_concat=dm_concat, **kwargs)

        if (not dm_concat and meta_size != size) or self.sg:
            print("In this mode, please make sure that meta_size and the wv dim are the same.")
            quit()

        self.meta_size = meta_size
        max_size = get_max_feature_size()
        if max_size is not None and self.meta_size > max_size:
            print("Meta feature size bigger than the allowed max of {}".format(max_size))
            print("Please increase this in meta_doc2vec_inner.pyx.")
            quit()
        self.dm_concat = dm_concat

        if self.dm and self.dm_concat:
            self.layer1_size = (self.dm_tag_count + (self.window_multiplier * self.window)) * self.vector_size + self.meta_size



        if documents is not None:
            self.build_vocab(documents, trim_rule=trim_rule)
            self.train(documents, total_examples=self.corpus_count, epochs=self.iter)

    def reset_weights(self):
        if self.dm and self.dm_concat:
            # expand l1 size to match concatenated tags+words length
            self.layer1_size = (self.dm_tag_count + (self.window_multiplier * self.window)) * self.vector_size + self.meta_size
            logger.info("using concatenative %d-dimensional layer1" % (self.layer1_size))
        super(Doc2Vec, self).reset_weights()
        self.docvecs.reset_weights(self)

    def _do_train_job(self, job, alpha, inits):
        work, neu1 = inits
        tally = 0
        for doc in job:
            indexed_doctags = self.docvecs.indexed_doctags(doc.tags)
            doctag_indexes, doctag_vectors, doctag_locks, ignored = indexed_doctags
            if self.sg:
                tally += train_meta_document_dbow(self, doc.words, doctag_indexes, doc.features, alpha, work,
                                                  train_words=self.dbow_words,
                                                  doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            elif self.dm_concat:
                tally += train_meta_document_dm_concat(self, doc.words, doctag_indexes, doc.features, alpha, work, neu1,
                                                        doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            else:
                tally += train_meta_document_dm(self, doc.words, doctag_indexes, doc.features, alpha, work, neu1,
                                                doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            self.docvecs.trained_item(indexed_doctags)
        return tally, self._raw_word_count(job)

    def infer_vector(self, doc_words, alpha=0.1, min_alpha=0.0001, steps=5):
        """
        Infer a vector for given post-bulk training document.

        Document should be a list of (word) tokens.
        """
        print("TODO?")
        raise NotImplementedError()
