# Natural language processing(NLP)
Steps involved in Natural Languagae Proceesing Algorithm
- Tokenize: Flattens the entire text into a basket of words
- Case Change(Optional based on applications): Converts same case to avoid having duplicate words in different case
- Stop Word Removal: Remove the common term or insignificant words from the bag
- Stemming: Keeps only the root words. i.e removing plurals, multiple tense forms etc
- Count Vectorization: Converts the bag of words into sparse matrix with frequency of each words from the bag. This step involves 'Term frequency-inverse document frequency (TF-IDF)'. TF-IDF determines the frequency of a word in the entire bag of words. This helps to determine the importance(weightage) of each word in the entire document
