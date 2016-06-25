import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.base import TransformerMixin



def test():
    target_label = [u'weather', u'audio',u'pic',u'calculate',u'music', u'poem']
    training_text_raw = []
    training_label = []
    with open ('./training_source.csv','r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if len(line) > 1 and line[1] in target_label:
                training_text_raw.append(unicode(line[0],"utf-8"))
                training_label.append(line[1])
        print training_label

        training_text = []
    for text in training_text_raw:
        seg_text = seg(text)
        training_text.append(seg_text)
    text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer(use_idf=False)),

                     ('clf', MultinomialNB()),
])

    scores = cross_validation.cross_val_score(text_clf, training_text, training_label, cv=8)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    text_clf.fit(training_text, training_label)

    while True:
        k_text = raw_input("\nPlease input:")
        if k_text == "exit":
            break
        print text_clf.predict([seg(unicode(k_text,'utf-8'))])


def seg(raw_text):
    assert isinstance(raw_text, unicode)
    return u' '.join(jieba.cut(raw_text))



if __name__ == '__main__':
    test()
