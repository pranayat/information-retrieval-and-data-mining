import pandas as pd
import math
import sys
import getopt

documents_df = pd.read_csv("./input/documents.csv",
    sep = ";",
    names = ["doc_id", "stemmed_content", "content"])
queries_df = pd.read_csv("./input/queries.csv",
    sep = ";",
    names = ["query_id", "stemmed_query", "query", "stemmed_keywords", "keywords"])
doc_count = documents_df.count().doc_id
optlist, args = getopt.getopt(sys.argv[1:], 'v:s:', [
    'variant=', 'stemming='])
variant = "classic",
stemming = False
doc_content_column = 3
query_content_column = 3
k = 5

for opt, opt_val in optlist:
    if opt in ('v', '--variant'):
        variant = opt_val
    if opt in ('s', '--stemming'):
        if opt_val in ['true', 'True']:
            stemming = True
            doc_content_column = 2
            query_content_column = 2

print('Computing...')
print('variant = ', variant)
print('stemming = ', stemming)

def get_doc_frequency():

    vocabulary = []
    for query in queries_df.itertuples():
        try:
            vocabulary += query[query_content_column].split()
        except AttributeError:
            continue
        except TypeError:
            continue

    vocabulary = list(dict.fromkeys(vocabulary))
    doc_frequency = {}
    for term in vocabulary:        
        doc_frequency[term] = 0
        for document in documents_df.itertuples():
            try:
                content = document[doc_content_column].split()
            except AttributeError:
                continue

            if term in content:
                doc_frequency[term] += 1

    return doc_frequency

def get_ranked_docs_by_query():

    tf_idf = {}
    results_by_query = {}
    for query in queries_df.itertuples():
            
        doc_frequency = get_doc_frequency()
        query_id = query[1]
        query_text = query[query_content_column]
        query_vocabulary = list(dict.fromkeys(query_text.split()))
        for document in documents_df.itertuples():

            doc_id = document[1]
            content = document[doc_content_column]
            tf_idf[doc_id] = 0
            for term in query_vocabulary:
                try:
                    tf = content.count(term)
                    if variant == 'log':
                        idf = math.log(doc_count / (1 + doc_frequency[term])) # adding 1 to avoid division by 0
                        tf_idf[doc_id] += tf * idf
                    elif variant == 'sublinear':
                        if tf > 0:
                            wf = 1 + math.log(tf)
                        else:
                            wf = 0 
                        idf = doc_count / (1 + doc_frequency[term])
                        tf_idf[doc_id] += wf * idf
                    elif variant == 'classic':
                        idf = doc_count / (1 + doc_frequency[term])
                        tf_idf[doc_id] += tf * idf
                    
                except AttributeError:
                    continue

        results_by_query[query_id] = sorted(tf_idf.items(), key = lambda kv: kv[1], reverse = True)[:k]

    result_set_dict = {}
    for query_id, query_results in results_by_query.items():
        result_set_dict[query_id] = set()
        for query_result in query_results:
            result_set_dict[query_id].add(query_result[0])

    return result_set_dict

def get_assessments_by_query():
    assessments_df = pd.read_csv("./input/assessments.csv",
        sep = ";",
        names = ["query_id", "doc_id"])
    assessment_set_dict = {}
    for assessment in assessments_df.itertuples():
        query_id = assessment[1]
        if query_id not in assessment_set_dict:
            assessment_set_dict[query_id] = set()
        else:
            assessment_set_dict[query_id].add(assessment[2])

    return assessment_set_dict

def get_recall_avg_precision_for_query(result_set, assessment_set):

    precision_sum = 0
    for i in range(1, k):
        tp_at_k = len(result_set.intersection(assessment_set)) # true positives in top m results
        fn_at_k = len(assessment_set.difference(result_set))
        if tp_at_k == 0: # if no relevant results fetched, precision is 0
            precision_at_k = 0
        else:            
            fp_at_k = len(result_set.difference(assessment_set))
            precision_at_k = tp_at_k/(tp_at_k + fp_at_k)
        
        precision_sum += precision_at_k
        avg_precision = precision_sum / k

        recall = tp_at_k/(tp_at_k + fn_at_k)

    return (recall, avg_precision)

def compute_recall_avg_precision():
    # list to store tuples (<query_id>, <recall>, <average_precision>)
    precision_avg_recall_list = []
    assessment_set_dict = get_assessments_by_query()
    result_set_dict = get_ranked_docs_by_query()
    for query_id, assessment_set in assessment_set_dict.items():
        precision_avg_recall = get_recall_avg_precision_for_query(result_set_dict[query_id], assessment_set_dict[query_id])
        precision_avg_recall_list.append((query_id, precision_avg_recall[0], precision_avg_recall[1]))

    print(precision_avg_recall_list)
    pd.DataFrame(precision_avg_recall_list,
        columns=["query_id", "recall", "average_precision"]).to_csv("./output/q1_1_"
            + variant + "_stemming-" + str(stemming) + "_.csv", sep = ",")

compute_recall_avg_precision()