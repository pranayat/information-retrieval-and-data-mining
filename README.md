# Folder Structure
```
./Assignment 2 - Code
  |__ input
    |__documents.csv
    |__assessments.csv
    |__queries.csv
  |__ output
    |__ q1_1_<variant>_stemming-<True/false>_.csv
  |__ q1.py
```

# Formulas
```
 tfidf(classic) = tf * |D|/(1 + df)
 tfidf(log) = tf * log|D|/(1 + df)
 tfidf(sublinear) = (1 + log(tf)) * |D|/(1 + df) if tf > 0 and 0 otherwise

 We are adding 1 to the denominator prevent division by 0 in case df = 0

 Recall for query q = tp / (tp + fn)
 Average Precision for query q = Sigma(Precision_at_k()) where k = 1 to number of relevant documents in the top k results
 
```
# Execution
```
python q1.py --variant=<variant> --stemming=<stemming>

--variant can take the following values: classic, log, sublinear
--stemming can take the following values: true, false

Eg.
python q1.py --variant=sublinear --stemming=true

The file will be written in ./output/q1_1_sublinear_stemming-True_.csv with the following structure:
query_id	recall	average_precision
```
