An Implementation of the TF-IDF algorithm to rank documents in importance. This implementation can be used for text summary. 

1. Text was cleaned:
- normalized (punctuations removed, transformed to lowercase).
- stop words removed.
- individual words lemmatized.

2. Calculated TF.

3. Calculated IDF.

4. Calculated TF-IDF.

5. Ranked documents in importance by summing TF-IDF values for the words it contained. 
