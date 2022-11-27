import math
import re

from bsbi import BSBIIndex, preprocess_text
from compression import VBEPostings
from letor import Data, LSIModel, Ranker

######## >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP


def rbp(ranking, p=0.8):
    """menghitung search effectiveness metric score dengan
    Rank Biased Precision (RBP)

    Parameters
    ----------
    ranking: List[int]
       vektor biner seperti [1, 0, 1, 1, 1, 0]
       gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
       Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
               di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
               di rank-6 tidak relevan

    Returns
    -------
    Float
      score RBP
    """
    score = 0.0
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking):
    """menghitung search effectiveness metric score dengan
    Discounted Cumulative Gain

    Parameters
    ----------
    ranking: List[int]
       vektor biner seperti [1, 0, 1, 1, 1, 0]
       gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
       Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
               di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
               di rank-6 tidak relevan

    Returns
    -------
    Float
      score DCG
    """
    # TODO
    score = 0.0
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] / math.log2(i + 1)
    return score


def ap(ranking):
    """menghitung search effectiveness metric score dengan
    Average Precision

    Parameters
    ----------
    ranking: List[int]
       vektor biner seperti [1, 0, 1, 1, 1, 0]
       gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
       Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
               di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
               di rank-6 tidak relevan

    Returns
    -------
    Float
      score AP
    """
    # TODO
    score = 0.0
    r_tally = 0
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        if ranking[pos]:
            r_tally += 1
            score += r_tally / i
    return score / r_tally


######## >>>>> memuat qrels


def load_qrels(qrel_file="qrels.txt", max_q_id=30, max_doc_id=1033):
    """memuat query relevance judgment (qrels)
    dalam format dictionary of dictionary
    qrels[query id][document id]

    dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
    relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
    Doc 10 tidak relevan dengan Q3.

    """
    qrels = {
        "Q" + str(i): {i: 0 for i in range(1, max_doc_id + 1)}
        for i in range(1, max_q_id + 1)
    }
    with open(qrel_file) as file:
        for line in file:
            parts = line.strip().split()
            qid = parts[0]
            did = int(parts[1])
            qrels[qid][did] = 1
    return qrels


######## >>>>> EVALUASI !

def eval(qrels, rank_args=None, query_file="queries.txt", k=1000):
    """
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
    """
    BSBI_instance = BSBIIndex(
        data_dir="collection", postings_encoding=VBEPostings, output_dir="index"
    )

    with open(query_file) as file:
        rbp_scores = []
        dcg_scores = []
        ap_scores = []

        for qline in file:
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            ranking = []
            if not rank_args:
                retrieved_data = BSBI_instance.retrieve_tfidf(query, k=k)
                scoring_method = "TF-IDF"
            else:
                retrieved_data = BSBI_instance.retrieve_bm25(query, *rank_args, k=k)
                scoring_method = "BM25"

            for (_, doc_path) in retrieved_data:
                did = int(re.search(r".*\\.*\\(.*)\.txt", doc_path).group(1))
                ranking.append(qrels[qid][did])

            rbp_scores.append(rbp(ranking))
            dcg_scores.append(dcg(ranking))
            ap_scores.append(ap(ranking))

    print(f"Hasil evaluasi ranked retrieval {scoring_method} terhadap 30 queries")
    print("RBP score =", sum(rbp_scores) / len(rbp_scores))
    print("DCG score =", sum(dcg_scores) / len(dcg_scores))
    print("AP score  =", sum(ap_scores) / len(ap_scores))


def eval_letor(qrels, rank_args=None, query_file="queries.txt", k=1000):
    # model lsi dan lgbm sudah di train di main letor.py
    lsi = LSIModel()
    lsi.load_model(1669482083)

    ranker = Ranker(lsi)
    ranker.load_ranker(1669482105)
    
    BSBI_instance = BSBIIndex(
        data_dir="collection", postings_encoding=VBEPostings, output_dir="index"
    )

    with open(query_file) as file:
        rbp_scores = []
        dcg_scores = []
        ap_scores = []

        for qline in file:
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            ranking = []
            if not rank_args:
                retrieved_data = BSBI_instance.retrieve_tfidf(query, k=k)
                scoring_method = "TF-IDF"
            else:
                retrieved_data = BSBI_instance.retrieve_bm25(query, *rank_args, k=k)
                scoring_method = "BM25"

            docs = []
            for (_, doc_path) in retrieved_data:
                did = int(re.search(r".*\\.*\\(.*)\.txt", doc_path).group(1))
                with open(doc_path) as f:
                    terms = preprocess_text(f.read())
                    doc = " ".join(terms)
                docs.append((did, doc))
            scores = ranker.predict(query, docs)
            sorted_did_scores = ranker.get_serp(docs, scores)

            for (did, _) in sorted_did_scores:
                ranking.append(qrels[qid][did])

            rbp_scores.append(rbp(ranking))
            dcg_scores.append(dcg(ranking))
            ap_scores.append(ap(ranking))

    print(f"Hasil evaluasi ranked retrieval {scoring_method} terhadap 30 queries")
    print("RBP score =", sum(rbp_scores) / len(rbp_scores))
    print("DCG score =", sum(dcg_scores) / len(dcg_scores))
    print("AP score  =", sum(ap_scores) / len(ap_scores))


if __name__ == "__main__":
    qrels = load_qrels()

    assert qrels["Q1"][166] == 1, "qrels salah"
    assert qrels["Q1"][300] == 0, "qrels salah"

    eval(qrels)
    eval(qrels, [2, 0.75])

    eval_letor(qrels)
    eval_letor(qrels, [2, 0.75])
