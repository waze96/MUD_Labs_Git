import nltk

#Tokenizer function. You can add here different preprocesses.
def preprocess(sentence, labels):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    # Place your code here
    # Keep in mind that sentence splitting affectes the number of sentences
    # and therefore, you should replicate labels to match.
    print(type(sentence))
    print(type(labels))
    print("AA")
    # Remove strange tokens
    tokensToRemove = ["[","]","《","》"]
    for t in tokensToRemove:
        sentence = sentence.replace(t, "")

    # Split sentence in more subsentences
    parts = sentence.split("，")
    return sentence,labels



