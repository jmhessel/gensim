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
import threading
import warnings

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from collections import namedtuple, defaultdict
from timeit import default_timer

from numpy import zeros, sum as np_sum, add as np_add, concatenate, \
    repeat as np_repeat, array, float32 as REAL, empty, ones, memmap as np_memmap, \
    sqrt, newaxis, ndarray, dot, vstack, dtype, divide as np_divide, integer


from gensim.utils import call_on_class_only
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.models.word2vec import Word2Vec, train_cbow_pair, train_sg_pair, train_batch_sg, score_cbow_pair, score_sg_pair, score_sentence_sg
from gensim.models.doc2vec import Doc2Vec, DocvecsArray, Doctag
from gensim.models.keyedvectors import KeyedVectors
from six.moves import xrange, zip
from six import string_types, integer_types

logger = logging.getLogger(__name__)

from gensim.models.meta_doc2vec_inner import train_meta_document_dm_concat, train_meta_document_dm, get_max_feature_size

try:
    from gensim.models.meta_doc2vec_inner import train_meta_document_dm_concat, train_meta_document_dm, train_meta_document_dbow, get_max_feature_size
    from gensim.models.meta_doc2vec_inner import score_meta_document_dbow, score_meta_document_dm, score_meta_document_dm_concat
    from gensim.models.word2vec_inner import FAST_VERSION  # blas-adaptation shared from word2vec
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


    def score_meta_document_dm_concat(model, doc_words, doctag_indexes, doc_features, work=None, neu1=None):
        """
        Obtain likelihood score for a single document in a fitted DBOW representaion.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2vec_inner instead.

        """
        log_prob_sentence = 0.0
        if model.negative:
            raise RuntimeError("scoring is only available for HS=True")

        doctag_vectors = model.docvecs.doctag_syn0
        word_vectors = model.wv.syn0

        doctag_len = len(doctag_indexes)

        word_vocabs = [model.wv.vocab[w] for w in doc_words if w in model.wv.vocab]

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
            l1 = concatenate((doctag_vectors[doctag_indexes], word_vectors[word_context_indexes])).ravel()
            l1 = concatenate((l1, doc_features))
            log_prob_sentence += score_cbow_pair(model, predict_word, l1)
        return log_prob_sentence


    def score_meta_document_dm(model, doc_words, doctag_indexes, doc_features, work=None, neu1=None):
        """
        Obtain likelihood score for a single document in a fitted CBOW representaion.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2vec_inner instead.

        """
        log_prob_sentence = 0.0
        if model.negative:
            raise RuntimeError("scoring is only available for HS=True")

        doctag_vectors = model.docvecs.doctag_syn0
        doctag_len = len(doctag_indexes)

        word_vocabs = [model.wv.vocab[w] for w in doc_words if w in model.wv.vocab]
        for pos, word in enumerate(word_vocabs):
            if word is None:
                continue  # OOV word in the input sentence => skip

            start = max(0, pos - model.window)
            end = (pos + model.window + 1) if not model.asymmetric_window else pos
            window_pos = enumerate(word_vocabs[start:end], start)
            word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]

            l1 = np_sum(model.wv.syn0[word2_indices], axis=0) + np_sum(doctag_vectors[doctag_indexes], axis=0) + doc_features
            count = len(word2_indices) + len(doctag_indexes) + 1.
            if model.cbow_mean and count > 1 :
                l1 /= count
            log_prob_sentence += score_cbow_pair(model, word, l1)
        return log_prob_sentence


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

        if not dm_concat and meta_size != size:
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

    def score(self, documents, total_documents=int(1e6), chunksize=100, queue_factor=2, report_delay=1):
        """
        Score the log probability for a sequence of documents (can be a once-only generator stream).
        Each document must must be a TaggedDocument
        This does not change the fitted model in any way (see Word2Vec.train() for that).

        We have currently only implemented score for the hierarchical softmax scheme,
        so you need to have run word2vec with hs=1 and negative=0 for this to work.

        Note that you should specify total_documents; we'll run into problems if you ask to
        score more than this number of sentences but it is inefficient to set the value too high.

        See the article by [taddy]_ and the gensim demo at [deepir]_ for examples of how to use such scores in document classification.

        .. [taddy] Taddy, Matt.  Document Classification by Inversion of Distributed Language Representations, in Proceedings of the 2015 Conference of the Association of Computational Linguistics.
        .. [deepir] https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/deepir.ipynb

        """
        if FAST_VERSION < 0:
            warnings.warn("C extension compilation failed, scoring will be slow. "
                          "Install a C compiler and reinstall gensim for fastness.")

        logger.info(
            "scoring documents with %i workers on %i vocabulary and %i features, "
            "using sg=%s hs=%s sample=%s and negative=%s",
            self.workers, len(self.wv.vocab), self.layer1_size, self.sg, self.hs, self.sample, self.negative)

        if not self.wv.vocab:
            raise RuntimeError("you must first build vocabulary before scoring new data")

        if not self.hs:
            raise RuntimeError("We have currently only implemented score \
                    for the hierarchical softmax scheme, so you need to have \
                    run word2vec with hs=1 and negative=0 for this to work.")

        def worker_loop():
            """Compute log probability for each sentence, lifting lists of sentences from the jobs queue."""
            while True:
                job = job_queue.get()
                if job is None:  # signal to finish
                    break
                ns = 0
                for sentence_id, sentence in job:
                    work = zeros(1, dtype=REAL)  # for sg hs, we actually only need one memory loc (running sum)
                    neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
                    indexed_doctags = self.docvecs.indexed_doctags(sentence.tags)
                    doctag_indexes, doctag_vectors, _, _ = indexed_doctags
                    doc_features = sentence.features

                    if sentence_id >= total_documents:
                        break
                    if self.sg:
                        score = score_meta_document_dbow(self, sentence.words, doctag_indexes, doc_features, work)
                    elif self.dm_concat:
                        if len(doctag_indexes) != len(sentence.tags):
                            for s in sentence.tags:
                                if s not in self.docvecs:
                                    print("Error: didn't see {} in training.".format(s))
                                    print("Giving up!")
                                    score = np.nan
                        else:
                            score = score_meta_document_dm_concat(self, sentence.words, doctag_indexes, doc_features, work, neu1)
                    else:
                        score = score_meta_document_dm(self, sentence.words, doctag_indexes, doc_features, work, neu1)
                    sentence_scores[sentence_id] = score
                    ns += 1
                progress_queue.put(ns)  # report progress

        start, next_report = default_timer(), 1.0
        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        sentence_count = 0
        sentence_scores = matutils.zeros_aligned(total_documents, dtype=REAL)

        push_done = False
        done_jobs = 0
        jobs_source = enumerate(utils.grouper(enumerate(documents), chunksize))

        # fill jobs queue with (id, sentence) job items
        while True:
            try:
                job_no, items = next(jobs_source)
                if (job_no - 1) * chunksize > total_documents:
                    logger.warning(
                        "terminating after %i sentences (set higher total_documents if you want more).",
                        total_documents)
                    job_no -= 1
                    raise StopIteration()
                logger.debug("putting job #%i in the queue", job_no)
                job_queue.put(items)
            except StopIteration:
                logger.info(
                    "reached end of input; waiting to finish %i outstanding jobs",
                    job_no - done_jobs + 1)
                for _ in xrange(self.workers):
                    job_queue.put(None)  # give the workers heads up that they can finish -- no more work!
                push_done = True
            try:
                while done_jobs < (job_no + 1) or not push_done:
                    ns = progress_queue.get(push_done)  # only block after all jobs pushed
                    sentence_count += ns
                    done_jobs += 1
                    elapsed = default_timer() - start
                    if elapsed >= next_report:
                        logger.info(
                            "PROGRESS: at %.2f%% sentences, %.0f sentences/s",
                            100.0 * sentence_count, sentence_count / elapsed)
                        next_report = elapsed + report_delay  # don't flood log, wait report_delay seconds
                else:
                    # loop ended by job count; really done
                    break
            except Empty:
                pass  # already out of loop; continue to next push

        elapsed = default_timer() - start
        self.clear_sims()
        logger.info(
            "scoring %i sentences took %.1fs, %.0f sentences/s",
            sentence_count, elapsed, sentence_count / elapsed)
        return sentence_scores[:sentence_count]
