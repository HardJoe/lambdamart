import datetime
import pickle
import random

import lightgbm as lgb
import numpy as np

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine


class Data:
    @staticmethod
    def parse_documents(doc_path: str) -> dict:
        documents = {}
        with open(doc_path) as file:
            for line in file:
                doc_id, content = line.split("\t")
                documents[doc_id] = content.split()
        return documents

    @staticmethod
    def parse_queries(q_path: str) -> dict:
        queries = {}
        with open(q_path) as file:
            for line in file:
                q_id, content = line.split("\t")
                queries[q_id] = content.split()
        return queries

    @staticmethod
    def parse_qrels(documents: dict, queries: dict, qrels_path: str) -> tuple:
        NUM_NEGATIVES = 1

        q_docs_rel = {}  # grouping by q_id terlebih dahulu
        with open(qrels_path) as file:
            for line in file:
                q_id, _, doc_id, rel = line.split("\t")
                if (q_id in queries) and (doc_id in documents):
                    if q_id not in q_docs_rel:
                        q_docs_rel[q_id] = []
                    q_docs_rel[q_id].append((doc_id, int(rel)))

        group_qid_count = []
        dataset = []

        for q_id in q_docs_rel:
            docs_rels = q_docs_rel[q_id]
            group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                dataset.append((queries[q_id], documents[doc_id], rel))
            # tambahkan satu negative (random sampling saja dari documents)
            dataset.append((queries[q_id], random.choice(list(documents.values())), 0))

        return group_qid_count, dataset


class LSIModel:
    NUM_LATENT_TOPICS = 200

    def __init__(self) -> None:
        self.dictionary = Dictionary()

    def make_model(self, documents: dict) -> None:
        """
        Bentuk dictionary, bag-of-words corpus, dan kemudian
        lakukan Latent Semantic Indexing dari kumpulan n dokumen.
        """
        bow_corpus = [
            self.dictionary.doc2bow(doc, allow_update=True)
            for doc in documents.values()
        ]
        self.model = LsiModel(bow_corpus, num_topics=self.NUM_LATENT_TOPICS)
        self.save_model(self.model)

    def save_model(self, model) -> None:
        file_name = f"model/lsa-{int(datetime.datetime.now().timestamp())}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump([model], f)

    def load_model(self, timestamp: int) -> None:
        file_name = f"model/lsa-{timestamp}.pkl"
        with open(file_name, "rb") as f:
            [self.model] = pickle.load(f)

    def parse_dataset(self, dataset: list):
        X = []
        Y = []

        for (query, doc, rel) in dataset:
            X.append(self.features(query, doc))
            Y.append(rel)

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    def features(self, query: str, doc: str) -> list:
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]

    def vector_rep(self, text: str) -> list:
        rep = [
            topic_value
            for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]
        ]
        if len(rep) == self.NUM_LATENT_TOPICS:
            return rep
        return [0.0] * self.NUM_LATENT_TOPICS


class Ranker:
    def __init__(self, model: LSIModel) -> None:
        self.model = model

    def make_ranker(self) -> None:
        self.ranker = lgb.LGBMRanker(
            objective="lambdarank",
            boosting_type="gbdt",
            n_estimators=100,
            importance_type="gain",
            metric="ndcg",
            num_leaves=40,
            learning_rate=0.02,
            max_depth=-1,
        )
        self.save_ranker(self.ranker)

    def save_ranker(self, ranker) -> None:
        file_name = f"model/lgbm-{int(datetime.datetime.now().timestamp())}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump([ranker], f)
    
    def load_ranker(self, timestamp: int) -> None:
        file_name = f"model/lgbm-{timestamp}.pkl"
        with open(file_name, "rb") as f:
            [self.ranker] = pickle.load(f)

    def fit(self, X, Y, group_qid_count: list) -> None:
        return self.ranker.fit(X, Y, group=group_qid_count, verbose=10)

    def predict(self, query, docs: list) -> list:
        X_unseen = []
        for _, doc in docs:
            X_unseen.append(self.model.features(query.split(), doc.split()))

        X_unseen = np.array(X_unseen)
        scores = self.ranker.predict(X_unseen)
        return scores

    def get_serp(self, docs: list, scores: list) -> list:
        did_scores = [x for x in zip([did for (did, _) in docs], scores)]
        sorted_did_scores = sorted(did_scores, key=lambda tup: tup[1], reverse=True)
        return sorted_did_scores


if __name__ == "__main__":
    test_query = "how much cancer risk can be avoided through lifestyle change ?"

    test_docs = [
        (
            "D1",
            "dietary restriction reduces insulin-like growth factor levels modulates apoptosis cell proliferation tumor progression num defici pubmed ncbi abstract diet contributes one-third cancer deaths western world factors diet influence cancer elucidated reduction caloric intake dramatically slows cancer progression rodents major contribution dietary effects cancer insulin-like growth factor igf-i lowered dietary restriction dr humans rats igf-i modulates cell proliferation apoptosis tumorigenesis mechanisms protective effects dr depend reduction multifaceted growth factor test hypothesis igf-i restored dr ascertain lowering igf-i central slowing bladder cancer progression dr heterozygous num deficient mice received bladder carcinogen p-cresidine induce preneoplasia confirmation bladder urothelial preneoplasia mice divided groups ad libitum num dr num dr igf-i igf-i/dr serum igf-i lowered num dr completely restored igf-i/dr-treated mice recombinant igf-i administered osmotic minipumps tumor progression decreased dr restoration igf-i serum levels dr-treated mice increased stage cancers igf-i modulated tumor progression independent body weight rates apoptosis preneoplastic lesions num times higher dr-treated mice compared igf/dr ad libitum-treated mice administration igf-i dr-treated mice stimulated cell proliferation num fold hyperplastic foci conclusion dr lowered igf-i levels favoring apoptosis cell proliferation ultimately slowing tumor progression mechanistic study demonstrating igf-i supplementation abrogates protective effect dr neoplastic progression",
        ),
        (
            "D2",
            "study hard as your blood boils",
        ),
        (
            "D3",
            "processed meats risk childhood leukemia california usa pubmed ncbi abstract relation intake food items thought precursors inhibitors n-nitroso compounds noc risk leukemia investigated case-control study children birth age num years los angeles county california united states cases ascertained population-based tumor registry num num controls drawn friends random-digit dialing interviews obtained num cases num controls food items principal interest breakfast meats bacon sausage ham luncheon meats salami pastrami lunch meat corned beef bologna hot dogs oranges orange juice grapefruit grapefruit juice asked intake apples apple juice regular charcoal broiled meats milk coffee coke cola drinks usual consumption frequencies determined parents child risks adjusted risk factors persistent significant associations children's intake hot dogs odds ratio num num percent confidence interval ci num num num hot dogs month trend num fathers intake hot dogs num ci num num highest intake category trend num evidence fruit intake provided protection results compatible experimental animal literature hypothesis human noc intake leukemia risk potential biases data study hypothesis focused comprehensive epidemiologic studies warranted",
        ),
        (
            "D4",
            "long-term effects calorie protein restriction serum igf num igfbp num concentration humans summary reduced function mutations insulin/igf-i signaling pathway increase maximal lifespan health span species calorie restriction cr decreases serum igf num concentration num protects cancer slows aging rodents long-term effects cr adequate nutrition circulating igf num levels humans unknown report data long-term cr studies num num years showing severe cr malnutrition change igf num igf num igfbp num ratio levels humans contrast total free igf num concentrations significantly lower moderately protein-restricted individuals reducing protein intake average num kg num body weight day num kg num body weight day num weeks volunteers practicing cr resulted reduction serum igf num num ng ml num num ng ml num findings demonstrate unlike rodents long-term severe cr reduce serum igf num concentration igf num igfbp num ratio humans addition data provide evidence protein intake key determinant circulating igf num levels humans suggest reduced protein intake important component anticancer anti-aging dietary interventions",
        ),
        (
            "D5",
            "cancer preventable disease requires major lifestyle abstract year num million americans num million people worldwide expected diagnosed cancer disease commonly believed preventable num num cancer cases attributed genetic defects remaining num num roots environment lifestyle lifestyle factors include cigarette smoking diet fried foods red meat alcohol sun exposure environmental pollutants infections stress obesity physical inactivity evidence cancer-related deaths num num due tobacco num num linked diet num num due infections remaining percentage due factors radiation stress physical activity environmental pollutants cancer prevention requires smoking cessation increased ingestion fruits vegetables moderate alcohol caloric restriction exercise avoidance direct exposure sunlight minimal meat consumption grains vaccinations regular check-ups review present evidence inflammation link agents/factors cancer agents prevent addition provide evidence cancer preventable disease requires major lifestyle",
        ),
    ]

    documents = Data.parse_documents("nfcorpus/train.docs")
    # test untuk melihat isi dari 2 dokumen
    print(documents["MED-329"])
    print(documents["MED-330"])

    queries = Data.parse_queries("nfcorpus/train.vid-desc.queries")
    # test untuk melihat isi dari 2 query
    print(queries["PLAIN-2428"])
    print(queries["PLAIN-2435"])

    group_qid_count, dataset = Data.parse_qrels(
        documents, queries, "nfcorpus/train.3-2-1.qrel"
    )
    # test
    print("number of Q-D pairs:", len(dataset))
    print("group_qid_count:", group_qid_count)
    assert sum(group_qid_count) == len(dataset), "ada yang salah"
    print(dataset[:2])

    lsi = LSIModel()
    lsi.make_model(documents)
    # test
    print(lsi.vector_rep(documents["MED-329"]))
    print(lsi.vector_rep(queries["PLAIN-2435"]))
    
    X, Y = lsi.parse_dataset(dataset)
    ranker = Ranker(lsi)
    ranker.make_ranker()
    ranker.fit(X, Y, group_qid_count)
    scores = ranker.predict(test_query, test_docs)
    sorted_did_scores = ranker.get_serp(test_docs, scores)

    # test SERP dari LGBMRanker
    print("Query        :", test_query)
    print("SERP/Ranking :")
    for (did, score) in sorted_did_scores:
        print(did, score)
