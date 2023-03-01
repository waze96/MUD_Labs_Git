import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import random
from utils import *
from classifiers import *
from preprocess import  preprocess

seed = 42
random.seed(seed)

# py ./LangDetect/source/langdetect.py --input C:\Users\jordi\Documents\GitHub\MUD_Labs_Git\LangDetect\data\dataset.csv --voc_size 2000 --analyzer 'char'

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", 
                        help="Input data in csv format", type=str)
    parser.add_argument("-v", "--voc_size", 
                        help="Vocabulary size", type=int)
    parser.add_argument("-a", "--analyzer",
                         help="Tokenization level: {word, char}", 
                        type=str, choices=['word','char'])
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    raw = raw[(raw['language']=='Japanese') | (raw['language']=='Chinese') | (raw['language']=='Thai') | (raw['language']=='Tamil') | (raw['language']=='Russian') | (raw['language']=='Hindi') | (raw['language']=='Korean')]
    # Languages
    languages = set(raw['language'])
    print('========')
    print('Languages', languages)
    print('========')

    # Split Train and Test sets
    X=raw['Text']
    y=raw['language']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    print('========')
    print('Split sizes:')
    print('Train:', len(X_train))
    print('Test:', len(X_test))
    print('========')
    
    # Preprocess text (Word granularity only)
    if args.analyzer == 'word':
        X_train, y_train = preprocess(X_train,y_train)
        X_test, y_test = preprocess(X_test,y_test)

    #Compute text features
    features, X_train_raw, X_test_raw = compute_features(X_train, 
                                                            X_test, 
                                                            analyzer=args.analyzer, 
                                                            max_features=args.voc_size)

    #dudu
    # UNI-GRAMS
    unigramVectorizer = CountVectorizer(analyzer='char', ngram_range=(1,1))
    X_unigram_train_raw = unigramVectorizer.fit_transform(X_train)
    X_unigram_test_raw = unigramVectorizer.transform(X_test)


    unigramFeatures = unigramVectorizer.get_feature_names()

    print('Number of unigrams in training set:', len(unigramFeatures))
    language_dict_unigram = train_lang_dict(X_unigram_train_raw.toarray(), y_train.values)
    relevantCharsPerLanguage = getRelevantCharsPerLanguage(unigramFeatures, language_dict_unigram, languages)
    
    # Print number of unigrams per language
    for lang in languages:    
        print(lang, len(relevantCharsPerLanguage[lang]))
    
    # get most common chars for a few European languages
    '''europeanLanguages = ['Portugese', 'Spanish', 'Latin', 'English', 'Dutch', 'Swedish']
    relevantChars_OnePercent = getRelevantCharsPerLanguage(unigramFeatures, language_dict_unigram, languages, 1e-2)

    # collect and sort chars
    europeanCharacters = []
    for lang in europeanLanguages:
        europeanCharacters += relevantChars_OnePercent[lang]
    europeanCharacters = list(set(europeanCharacters))
    europeanCharacters.sort()

    # build data
    indices = [unigramFeatures.index(f) for f in europeanCharacters]
    data = []
    for lang in europeanLanguages:
        data.append(language_dict_unigram[lang][indices])

    #build dataframe
    df = pd.DataFrame(np.array(data).T, columns=europeanLanguages, index=europeanCharacters)
    df.index.name = 'Characters'
    df.columns.name = 'Languages'
    '''

    # plot heatmap
    '''import seaborn as sn
    import matplotlib.pyplot as plt
    sn.set(font_scale=0.8) # for label size
    sn.set(rc={'figure.figsize':(10, 10)})
    sn.heatmap(df, cmap="Greens", annot=True, annot_kws={"size": 12}, fmt='.0%')# font size
    plt.show()'''
    
    # BI-GRAMS
    # number of bigrams
    '''bigramVectorizer = CountVectorizer(analyzer='char', ngram_range=(2,2))
    X_bigram_raw = bigramVectorizer.fit_transform(X_train)
    bigramFeatures = bigramVectorizer.get_feature_names()
    print('Number of bigrams', len(bigramFeatures))
    
    # top bigrams (>1%) for Spanish, Italian (Latin), English, Dutch, Chinese, Japanese, Korean
    language_dict_bigram = train_lang_dict(X_bigram_raw.toarray(), y_train.values)
    relevantCharsPerLanguage = getRelevantCharsPerLanguage(bigramFeatures, language_dict_bigram, languages, significance=1e-2)
    print('Spanish', relevantCharsPerLanguage['Spanish'])
    print('Italian (Latin)', relevantCharsPerLanguage['Latin'])
    print('English', relevantCharsPerLanguage['English'])
    print('Dutch', relevantCharsPerLanguage['Dutch'])
    print('Chinese', relevantCharsPerLanguage['Chinese'])
    print('Japanese', relevantCharsPerLanguage['Japanese'])'''

    # Mixture of Uni-Gram & Bi-Grams
    # Uni- & Bi-Gram Mixture CountVectorizer for top 1% features
    top1PrecentMixtureVectorizer = CountVectorizer(analyzer='char', ngram_range=(1,2), min_df=1e-2)
    X_top1Percent_train_raw = top1PrecentMixtureVectorizer.fit_transform(X_train)
    X_top1Percent_test_raw = top1PrecentMixtureVectorizer.transform(X_test)

    language_dict_top1Percent = train_lang_dict(X_top1Percent_train_raw.toarray(), y_train.values)

    top1PercentFeatures = top1PrecentMixtureVectorizer.get_feature_names()
    print('Length of features', len(top1PercentFeatures))
    print('')

    #Unique features per language
    relevantChars_Top1Percent = getRelevantCharsPerLanguage(top1PercentFeatures, language_dict_top1Percent, languages, 1e-5)
    for lang in relevantChars_Top1Percent:
        print("{}: {}".format(lang, len(relevantChars_Top1Percent[lang])))

    # Take top 50 (Uni- & Bi-Grams) per language
    top50PerLanguage_dict = getRelevantGramsPerLanguage(top1PercentFeatures, language_dict_top1Percent, languages)

    # top50
    allTop50 = []
    for lang in top50PerLanguage_dict:
        allTop50 += set(top50PerLanguage_dict[lang])

    top50 = list(set(allTop50))

    print()    
    print('All items:', len(allTop50))
    print('Unique items:', len(top50))

    relevantColumnIndices = getRelevantColumnIndices(np.array(top1PercentFeatures), top50)

    X_top50_train_raw = np.array(X_top1Percent_train_raw.toarray()[:,relevantColumnIndices])
    X_top50_test_raw = X_top1Percent_test_raw.toarray()[:,relevantColumnIndices] 

    print()
    print('train shape', X_top50_train_raw.shape)
    print('test shape', X_top50_test_raw.shape)
    #dudu

    print('========')
    print('Number of tokens in the vocabulary:', len(features))
    print('Coverage: ', compute_coverage(features, X_test.values, analyzer=args.analyzer))
    print('========')


    #Apply Classifier  
    X_train, X_test = normalizeData(X_train_raw, X_test_raw)
    y_predict = applyNaiveBayes(X_train, y_train, X_test)

    X_unigram_train, X_unigram_test = normalizeData(X_unigram_train_raw, X_unigram_test_raw)
    y_predict_nb_unigram = applyNaiveBayes(X_unigram_train, y_train, X_unigram_test)

    X_top1Percent_train, X_top1Percent_test = normalizeData(X_top1Percent_train_raw, X_top1Percent_test_raw)
    y_predict_nb_top1Percent = applyNaiveBayes(X_top1Percent_train, y_train, X_top1Percent_test)
    
    print('========')
    print('Prediction Results:')    
    plot_F_Scores(y_test, y_predict)
    plot_F_Scores(y_test, y_predict_nb_unigram)
    plot_F_Scores(y_test, y_predict_nb_top1Percent)
    print('========')
    
    plot_Confusion_Matrix(y_test, y_predict, "Greens") 
    plot_Confusion_Matrix(y_test, y_predict_nb_unigram, "Oranges")
    plot_Confusion_Matrix(y_test, y_predict_nb_top1Percent, "Reds")


    #Plot PCA
    print('========')
    print('PCA and Explained Variance:') 
    plotPCA(X_train, X_test,y_test, languages,X_unigram_train_raw) 
    plotPCA(X_train, X_test,y_test, languages) 
    plotPCA(X_unigram_train, X_unigram_test,y_test, languages) 
    plotPCA(X_top1Percent_train, X_top1Percent_test,y_test, languages) 
    print('========')
