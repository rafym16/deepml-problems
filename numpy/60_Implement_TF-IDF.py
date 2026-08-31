import numpy as np


def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents.

    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: List of lists containing TF-IDF scores for the query words in each document
    """
    if not corpus:
        return []

    df_map = {}

    for word in query:
        count = 0
        for doc in corpus:
            if word in doc:
                count += 1
            df_map[word] = count

    idf_map = {}
    for word in query:
        df = df_map[word]
        idf_map[word] = np.log(((len(corpus) + 1) / (df + 1))) + 1

    result = []
    for doc in corpus:
        doc_tf_idf = []
        doc_length = len(doc)
        for word in query:
            if doc_length == 0:
                tf = 0.0
            else:
                word_count = doc.count(word)
                tf = word_count / doc_length

            idf = idf_map[word]
            tf_idf = tf * idf
            doc_tf_idf.append(round(tf_idf, 5).tolist())
        result.append(doc_tf_idf)

    return result