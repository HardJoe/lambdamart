import os
import pickle
import contextlib
import heapq
import math
import re

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from tqdm import tqdm

# Collaborators:
# 1906308500 - Matthew Tumbur Parluhutan S. Siregar
# 1906350912 - Hendrico Kristiawan
# 1906398351 - Beltsazar Anugrah Sotardodo


class Weighting:
    @staticmethod
    def get_log_tf_weight(doc_tf):
        if doc_tf == 0:
            return 0
        return 1 + math.log10(doc_tf)

    @staticmethod
    def get_aug_tf_weight(doc_tf, max_tf):
        return 0.5 + 0.5 * doc_tf / max_tf

    @staticmethod
    def get_idf_weight(total_docs, term_df):
        return math.log10(total_docs / term_df)

    @staticmethod
    def get_prob_idf_weight(total_docs, term_df):
        return max(0.0, math.log((total_docs - term_df) / term_df))

    @staticmethod
    def get_bm25_tf_weight(k1, b, doc_tf, dl, avdl):
        return ((k1 + 1) * doc_tf) / (k1 * (1 - b + b * dl / avdl) + doc_tf)
    

def preprocess_text(text):
    """
    Digunakan sebelum memproses query saat retrieval dan document saat
    indexing.
    """
    # TODO
    text = text.lower()
    text = re.sub("\s+", " ", text)  # Remove excess whitespace
    text = re.sub("[^\w\s]", " ", text)  # Remove punctuations
    text = re.sub(r"\d+", "", text)  # Remove numbers

    text = word_tokenize(text)

    stops = set(stopwords.words("english"))
    text = [word for word in text if word not in stops]

    stemmer = SnowballStemmer("english")
    text = [stemmer.stem(word) for word in text]

    return text


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(
        self, data_dir, output_dir, postings_encoding, index_name="main_index"
    ):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, "terms.dict"), "wb") as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, "docs.dict"), "wb") as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, "terms.dict"), "rb") as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, "docs.dict"), "rb") as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        # TODO
        td_pairs = []

        for filename in os.listdir(os.path.join("collection", block_dir_relative)):
            doc_path = os.path.join("collection", block_dir_relative, filename)
            doc_id = self.doc_id_map[doc_path]

            with open(doc_path) as f:
                terms = preprocess_text(f.read())
                for term in terms:
                    term_id = self.term_id_map[term]
                    td_pairs.append((term_id, doc_id))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # TODO
        term_dict = {}

        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = dict()
            try:
                term_dict[term_id][doc_id] += 1
            except KeyError:
                term_dict[term_id][doc_id] = 1

        f = open("index_log.txt", "w")

        for term_id in sorted(term_dict.keys()):
            sorted_tf = sorted(term_dict[term_id].items(), key=lambda kv: kv[0])
            postings_list = [t[0] for t in sorted_tf]
            tf_list = [t[1] for t in sorted_tf]
            output = f"{term_id:<5} {self.term_id_map[term_id]:21} {postings_list} {tf_list} \n"
            f.write(output)
            index.append(term_id, postings_list, tf_list)

        f.close()

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(
                    list(zip(postings, tf_list)), list(zip(postings_, tf_list_))
                )
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan:
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = preprocess_text(query)
        if not terms:
            return []

        with InvertedIndexReader(
            self.index_name, self.postings_encoding, directory=self.output_dir
        ) as invert_map:
            total_docs = len(invert_map.doc_length)
            scores = [[0, i] for i in range(total_docs)]

            for t in terms:
                if t not in self.term_id_map:
                    continue
                t_id = self.term_id_map[t]
                idf_weight = Weighting.get_idf_weight(
                    total_docs, invert_map.postings_dict[t_id][1]
                )
                postings_list, tf_list = invert_map.get_postings_list(t_id)
                for i, doc_id in enumerate(postings_list):
                    tf_weight = Weighting.get_log_tf_weight(tf_list[i])
                    scores[doc_id][0] += tf_weight * idf_weight

        scores = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
        scores = [(score, self.doc_id_map[doc_id]) for [score, doc_id] in scores]
        return scores

    def retrieve_tfidf_apnbnn(self, query, k=10):
        """
        Retrieval TaaT dengan varian TF-IDF apn.bnn
        """
        # TODO
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = preprocess_text(query)
        if not terms:
            return []

        with InvertedIndexReader(
            self.index_name, self.postings_encoding, directory=self.output_dir
        ) as invert_map:
            total_docs = len(invert_map.doc_length)
            scores = [[0, i] for i in range(total_docs)]

            for t in terms:
                if t not in self.term_id_map:
                    continue
                t_id = self.term_id_map[t]
                idf_weight = Weighting.get_prob_idf_weight(
                    total_docs, invert_map.postings_dict[t_id][1]
                )
                postings_list, tf_list = invert_map.get_postings_list(t_id)
                for i, doc_id in enumerate(postings_list):
                    tf_weight = Weighting.get_aug_tf_weight(tf_list[i], max(tf_list))
                    scores[doc_id][0] += tf_weight * idf_weight

        scores = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
        scores = [(score, self.doc_id_map[doc_id]) for [score, doc_id] in scores]
        return scores

    def retrieve_tfidf_ntnlnn(self, query, k=10):
        """
        Retrieval TaaT dengan varian TF-IDF ntn.lnn

        """
        # TODO
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = preprocess_text(query)
        if not terms:
            return []

        with InvertedIndexReader(
            self.index_name, self.postings_encoding, directory=self.output_dir
        ) as invert_map:
            total_docs = len(invert_map.doc_length)
            scores = [[0, i] for i in range(total_docs)]

            for t in terms:
                if t not in self.term_id_map:
                    continue
                query_tf = terms.count(t)
                query_weight = Weighting.get_log_tf_weight(query_tf)

                t_id = self.term_id_map[t]
                idf_weight = Weighting.get_idf_weight(
                    total_docs, invert_map.postings_dict[t_id][1]
                )
                postings_list, tf_list = invert_map.get_postings_list(t_id)
                for i, doc_id in enumerate(postings_list):
                    doc_tf_weight = tf_list[i]
                    scores[doc_id][0] += doc_tf_weight * idf_weight * query_weight

        scores = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
        scores = [(score, self.doc_id_map[doc_id]) for [score, doc_id] in scores]
        return scores

    def retrieve_bm25(self, query, k1, b, k=10):
        """
        Retrieval TaaT dengan metode Okapi BM25
        """
        # TODO
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = preprocess_text(query)
        if not terms:
            return []

        with InvertedIndexReader(
            self.index_name, self.postings_encoding, directory=self.output_dir
        ) as invert_map:
            total_docs = len(invert_map.doc_length)
            scores = [[0, i] for i in range(total_docs)]

            for t in terms:
                if t not in self.term_id_map:
                    continue
                t_id = self.term_id_map[t]
                query_weight = Weighting.get_idf_weight(
                    total_docs, invert_map.postings_dict[t_id][1]
                )
                postings_list, tf_list = invert_map.get_postings_list(t_id)
                for i, doc_id in enumerate(postings_list):
                    doc_weight = Weighting.get_bm25_tf_weight(
                        k1,
                        b,
                        tf_list[i],
                        invert_map.doc_length[doc_id],
                        invert_map.avg_doc_length,
                    )
                    scores[doc_id][0] += doc_weight * query_weight

        scores = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
        scores = [(score, self.doc_id_map[doc_id]) for [score, doc_id] in scores]
        return scores

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = "intermediate_index_" + block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(
                index_id, self.postings_encoding, directory=self.output_dir
            ) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(
            self.index_name, self.postings_encoding, directory=self.output_dir
        ) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [
                    stack.enter_context(
                        InvertedIndexReader(
                            index_id, self.postings_encoding, directory=self.output_dir
                        )
                    )
                    for index_id in self.intermediate_indices
                ]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    nltk.download("stopwords")
    BSBI_instance = BSBIIndex(
        data_dir="collection", postings_encoding=VBEPostings, output_dir="index"
    )
    BSBI_instance.index()  # memulai indexing!
    print("indexing selesai")
