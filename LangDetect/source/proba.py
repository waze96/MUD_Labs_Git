




dic_f1_score_optimitzation = {}
dic_f1_score_optimitzation['1gram'] = {}
dic_f1_score_optimitzation['2gram'] = {}
dic_f1_score_optimitzation['1-2gram'] = {}
dic_f1_score_optimitzation['1-2-3gram'] = {}
dic_f1_score_optimitzation['1gram']['ngram_range'] = (1,1)
dic_f1_score_optimitzation['2gram']['ngram_range'] = (2,2)
dic_f1_score_optimitzation['1-2gram']['ngram_range'] = (1,2)
dic_f1_score_optimitzation['1-2-3gram']['ngram_range'] = (1,3)

for type_gram in dic_f1_score_optimitzation.keys():
    for max_features in [150, 250, 500, 1000, 2000, 4000, 6816,10000, 15000,25000]:

        # list with all the languages (unique)
        languages = set(raw['language']) 

        # Split Train and Test sets
        X=raw['Text']
        y=raw['language']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        # Vectorize the text
        unigramVectorizer = CountVectorizer(analyzer=analyzer, max_features=max_features, ngram_range=dic_f1_score_optimitzation[type_gram]['ngram_range'])
        X_unigram_train_raw = unigramVectorizer.fit_transform(X_train)
        X_unigram_test_raw = unigramVectorizer.transform(X_test)
        unigramFeatures = unigramVectorizer.get_feature_names()
        print('length of the vector:' + str(len(unigramFeatures)))

        # change variables names
        features = unigramFeatures      # features names
        X_train_raw = X_unigram_train_raw
        X_test_raw = X_unigram_test_raw

        compute_coverage(features, X_test.values, analyzer=analyzer)

        X_train, X_test = normalizeData(X_train_raw, X_test_raw)

        #Apply Classifier  
        y_predict = applyNaiveBayes(X_train, y_train, X_test)
 
        dic_f1_score_optimitzation[type_gram]['f1score'] = f1_score(y_test, y_predict, average='weighted')

