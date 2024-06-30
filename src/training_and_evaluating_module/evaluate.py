import os
import logging
from tqdm import trange
import heapq

import numpy as np
import torch
import evaluate

from data_module.utils import VLSP_NER_TAGS

logger = logging.getLogger(__name__)

ner_label_list = VLSP_NER_TAGS

def get_ner_labels(predictions, references, args):
    # Transform predictions and references tensos to numpy arrays
    if predictions.device.type == 'cpu' and references.device.type == 'cpu':
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()
    # Remove ignored index (special tokens)
    true_predictions = [
        [ner_label_list[p] for (p, l) in zip(pred, gold_label) if l != args.dummy_label_id]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
        [ner_label_list[l] for (p, l) in zip(pred, gold_label) if l != args.dummy_label_id]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    return true_predictions, true_labels

def compute_metrics_ner(metric, predictions=None, labels=None, args=None):
    if metric is not None:    
        results = metric.compute()
    else:
        assert predictions is not None and labels is not None, 'Predictions and labels can\'t be None!' 
        true_predictions, true_labels = get_ner_labels(
            predictions=predictions,
            references=labels,
            args=args
        )
        results = metric.compute(predictions=true_predictions, references=true_labels)
    return results

def get_true_mlm_result(predictions, references, args):
    # Transform predictions and references tensos to numpy arrays
    if predictions.device.type == 'cpu' and references.device.type == 'cpu':
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()
    # Remove ignored index (special tokens)
    true_predictions = [
        [ner_label_list[p] for (p, l) in zip(pred, gold_label) if l != args.dummy_label_id]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
        [ner_label_list[l] for (p, l) in zip(pred, gold_label) if l != args.dummy_label_id]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    return true_predictions, true_labels

def compute_metrics_ner(metric, predictions=None, labels=None, args=None):
    if metric is not None:    
        results = metric.compute()
    else:
        assert predictions is not None and labels is not None, 'Predictions and labels can\'t be None!' 
        true_predictions, true_labels = get_ner_labels(
            predictions=predictions,
            references=labels,
            args=args
        )
        results = metric.compute(predictions=true_predictions, references=true_labels)
    return results

class InformationRetrievalEvaluator():

    def __init__(
            self,
            queries: dict,
            corpus: dict,
            relevant_docs: dict,
            args,
            score_functions: dict,
            write_csv=True,
            name=None
    ):
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]
        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]
        self.relevant_docs = relevant_docs
        self.args = args
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))

        if name:
            self.name = "_" + name
        self.csv_file = "Information-Retrieval_evaluation" + self.name + "_results.csv"
        self.csv_headers = ['epoch', 'steps']

        for score_name in self.score_function_names:
            for k in self.args.accuracy_at_k:
                self.csv_headers.append("{}-Accuracy@{}".format(score_name, k))

            for k in self.args.precision_recall_at_k:
                self.csv_headers.append("{}-Precision@{}".format(score_name, k))
                self.csv_headers.append("{}-Recall@{}".format(score_name, k))

            for k in self.args.mrr_at_k:
                self.csv_headers.append("{}-MRR@{}".format(score_name, k))

            for k in self.args.ndcg_at_k:
                self.csv_headers.append("{}-NDCG@{}".format(score_name, k))

            for k in self.args.map_at_k:
                self.csv_headers.append("{}-MAP@{}".format(score_name, k))

    def __call__(self, model, epoch=-1, steps=-1):
        if epoch != -1:
            out_txt = (f' after epoch {epoch}' if steps == -1 
                       else f' in epch{epoch} after {steps} steps')
        else:
            out_txt = ':'
        logger.info('Information Retrieval Evaluation on ' + self.name + ' dataset' + out_txt)

        scores, queries_result_list = self.compute_metrices(model)

        # Write results to disk
        if self.args.output_dir is not None and self.write_csv:
            csv_path = os.path.join(self.args.output_dir, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode='w', encoding='utf-8')
                fOut.write(','.join(self.csv_headers))
                fOut.write('\n')
            else:
                fOut = open(csv_path, mode='a', encoding='utf-8')
            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.args.accuracy_at_k:
                    output_data.append(scores[name]["accuracy@k"][k])
                for k in self.args.precision_recall_at_k:
                    output_data.append(scores[name]["precision@k"][k])
                    output_data.append(scores[name]["recall@k"][k])
                for k in self.args.mrr_at_k:
                    output_data.append(scores[name]["mrr@k"][k])
                for k in self.args.ndcg_at_k:
                    output_data.append(scores[name]["ndcg@k"][k])
                for k in self.args.map_at_k:
                    output_data.append(scores[name]["map@k"][k])

            fOut.write(','.join(map(str, output_data)))
            fOut.write('\n')
            fOut.close()

        if self.main_score_function is None:
            return scores, queries_result_list
        else:
            return scores[self.main_score_function]["map@k"][max(self.map_at_k)]

    def compute_metrices(
            self,
            model,
            corpus_model=None,
            corpus_embeddings: torch.Tensor=None
    ):
        if corpus_model is None:
            corpus_model = model

        max_k = max(
            max(self.args.mrr_at_k),
            max(self.args.ndcg_at_k),
            max(self.args.accuracy_at_k),
            max(self.args.precision_recall_at_k),
            max(self.args.map_at_k),
        )

        # Compute embeddings for queries
        query_embeddings = model.encode_query(self.queries)

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(self.queries))]
        
        # Iterate over chunks of the corpus
        for corpus_start_idx in trange(
            0, len(self.corpus), self.args.corpus_chunk_size, desc='Corpus Chunk'
        ):
            corpus_end_idx = min(corpus_start_idx + self.args.corpus_chunk_size, len(self.corpus))

            # Encode chunk of corpus
            if corpus_embeddings is None:
                sub_corpus_embeddings = corpus_model.encode_doc(self.corpus[corpus_start_idx: corpus_end_idx])
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx: corpus_end_idx]

            # Compute similarities!
            for name, score_function in self.score_functions.items():
                # return query_embeddings, sub_corpus_embeddings
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)
                # Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                    pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False
                )
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(self.queries)):
                    for sub_corpus_id, score in zip(
                        pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]
                    ):
                        corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                        if len(queries_result_list[name][query_itr]) < max_k:
                            heapq.heappush(
                                queries_result_list[name][query_itr], (score, corpus_id)
                            )  # heaqp tracks the quantity of the first element in the tuple
                        else:
                            heapq.heappushpop(queries_result_list[name][query_itr], (score, corpus_id))
        
        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {'corpus_id': corpus_id, 'score': score}
        
        logger.info(f'Queries: {len(self.queries)}')
        logger.info(f'Corpus: {len(self.corpus)}\n')

        # Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}
        # Output
        for name in self.score_function_names:
            logger.info(f'Score-Function: {name}')
            self.output_scores(scores[name])

        return scores, queries_result_list

    def compute_metrics(self, queries_result_list):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.args.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.args.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.args.precision_recall_at_k}
        MRR = {k: 0 for k in self.args.mrr_at_k}
        ndcg = {k: [] for k in self.args.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.args.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is across the top-k documents
            for k_val in self.args.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.args.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.args.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.args.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(
                    true_relevances, k_val
                )
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.args.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        result = {
            "accuracy@k": num_hits_at_k,
            "precision@k": precisions_at_k,
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
            "map@k": AveP_at_k,
        }
        return result
    
    def output_scores(self, scores):
        for k in scores["accuracy@k"]:
            logger.info("Accuracy@{}: {:.2f}%".format(k, scores["accuracy@k"][k] * 100))

        for k in scores["precision@k"]:
            logger.info("Precision@{}: {:.2f}%".format(k, scores["precision@k"][k] * 100))

        for k in scores["recall@k"]:
            logger.info("Recall@{}: {:.2f}%".format(k, scores["recall@k"][k] * 100))

        for k in scores["mrr@k"]:
            logger.info("MRR@{}: {:.4f}".format(k, scores["mrr@k"][k]))

        for k in scores["ndcg@k"]:
            logger.info("NDCG@{}: {:.4f}".format(k, scores["ndcg@k"][k]))

        for k in scores["map@k"]:
            logger.info("MAP@{}: {:.4f}".format(k, scores["map@k"][k]))

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg