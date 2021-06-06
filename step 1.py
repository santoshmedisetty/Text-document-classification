import glob
import os
import xml.etree.ElementTree as ET
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

DataFrameColumns=['Headline','Text','Bip_Topics','Date_Published','Item_ID','File_Name']
biptopiclist=[]
rows=[]
DataFrameCustom=[]
stopwords = set(stopwords.words('english'))
def dataextraction():
    for file in glob.iglob('/Users/santoshkumarmedisetty/Downloads/ML assignment/Data/*/*.xml'):
        bip_topic = []
        mytree = ET.parse(file)
        myroot = mytree.getroot()
        path, filename = os.path.split(file)
        itemid = myroot.attrib.get('itemid')
        hl = myroot.find('headline').text
        dc = myroot.findall("./metadata/dc[@element='dc.date.published']")
        datepublished = dc[0].attrib.get("value")
        paragraph = ""
        textnode = myroot.find("text")
        for node in textnode:
            paragraph = paragraph + node.text

        LemmatizedSentence=stemlemm(paragraph)

        Sentencewostopwords=stopword(LemmatizedSentence)
        Sentencewostopwords=' '.join(Sentencewostopwords)
        cd = myroot.findall("./metadata/codes[@class='bip:topics:1.0']/code")

        for i in cd:
            k = i.attrib.get("code")
            biptopiclist.append(k)
            rows.append(
                {"Headline": hl, "Text": Sentencewostopwords, "Bip_Topics": k, "Date_Published": datepublished, "Item_ID": itemid,
                 "File_Name": filename})
            break

    DataFrameCustom = pd.DataFrame(rows, columns=DataFrameColumns)
    return DataFrameCustom


def stopword(paragraph):
    words_tokens = word_tokenize(paragraph)
    sentence = []
    for w in words_tokens:
        if w not in stopwords:
            sentence.append(w)
    return sentence
def uniquebipcodes(biptopiclist):
    unique_codes=[]
    for code in biptopiclist:
        if code not in unique_codes:
            unique_codes.append(code)
    print(unique_codes)
    return unique_codes

def featureextraction(FirstDF):
    vectorizer = CountVectorizer()
    textdata = FirstDF["Text"]
    bip = FirstDF["Bip_Topics"]
    biparray = pd.Series(bip).values
    filename = FirstDF["File_Name"]
    FeatureData = vectorizer.fit_transform(textdata).toarray()
    FeatureDataFrameData = np.column_stack((FeatureData, biparray))
    dataFrameColumns = vectorizer.get_feature_names()
    dataFrameColumns.insert(len(dataFrameColumns), 'Labels')
    FeatureDataFrame = pd.DataFrame(data=FeatureDataFrameData, columns=dataFrameColumns, index=filename)
    return FeatureDataFrame

def stemlemm(sentence):
    words_tokens = word_tokenize(sentence)
    filtered_sentence_list = [w for w in words_tokens if w not in stopwords]

    StemSentence = []
    LemmSentence = []
    Stemmed_Sentence = []
    for word in filtered_sentence_list:
        StemSentence.append(PorterStemmer().stem(word))

    for wor in StemSentence:
        LemmSentence.append(WordNetLemmatizer().lemmatize(wor))

    Stemmed_Sentence = " ".join(StemSentence)
    Lemmatized_Sentence = " ".join(LemmSentence)

    return Lemmatized_Sentence

def kfoldsplit(DF):
    targetlabel = DF['Labels']
    textlabel=DF.iloc[:, :-1].values
    kf=KFold(n_splits=2)
    for train_index, test_index in kf.split(textlabel,targetlabel):
        textlabel_train, textlabel_test, targetlabel_train, targetlabel_test = textlabel[train_index], textlabel[test_index], targetlabel[train_index], targetlabel[test_index]

    print(textlabel_train)
    print(textlabel_test)
    print(targetlabel_train)
    print(targetlabel_test)

# KFold approach is the best approach because it makes sure that every observation is used both in testing and training set.
#It is  best if we have limited data. It provides less bias results. The disadvantage of this approach is, it takes more computations(k computations)
# which is a disadvantage if the dataset is large

def classifier(DF):
    X = DF.iloc[:, :-1].values
    y = DF['Labels']

    kf = KFold(n_splits=2)
    for train, test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

    # KNN
    KNN_model = KNeighborsClassifier(n_neighbors=10)
    KNN_model.fit(X_train, y_train)
    KNN_prediction = KNN_model.predict(X_test)
    print("Metrics of KNN")
    qualityofclassifier(KNN_prediction, y_test)


def trainclassification(DF):

    X = DF.iloc[:, :-1].values
    y = DF['Labels']

    kf = KFold(n_splits=2)
    for train,test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        # print(X_train)
        # print(X_test)
        # print(y_train)
        # print(y_test)

    #Normalisation
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # PCA
    pca = PCA(n_components=70)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print(X_train)


    #KNN
    KNN_model = KNeighborsClassifier(n_neighbors=10)
    KNN_model.fit(X_train, y_train)
    KNN_prediction = KNN_model.predict(X_test)
    print("Metrics of KNN")
    qualityofclassifier(KNN_prediction,y_test)

    # #SVM
    SVC_model = svm.SVC(kernel='linear', C=0.1)
    SVC_model.fit(X_train, y_train)
    SVC_prediction = SVC_model.predict(X_test)
    print("Metrics of SVM are")
    qualityofclassifier(SVC_prediction,y_test)

    #RandomForest
    RF = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0)
    RF.fit(X_train, y_train)
    RF_Prediction=RF.predict(X_test)
    print("Metrics of Random Forest are")
    qualityofclassifier(RF_Prediction,y_test)


    #DecisionTreeClassifier
    DT_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=10)
    DT_gini.fit(X_train, y_train)
    DT_pred = DT_gini.predict(X_test)
    print("Metrics of DT are")
    qualityofclassifier(DT_pred, y_test)

    #Linear Regression
    # LRegressor = LinearRegression()
    # LRegressor.fit(X_train, y_train)
    # LRegressor_pred = LRegressor.predict(X_test)
    # qualityofclassifier(LRegressor_pred,y_test)

    #NeuralNetwork
    p = MLPClassifier(random_state=42,tol=0.001,max_iter=500)
    p.fit(X_train,y_train)
    NN_Pred=p.predict(X_test)
    print("Metrics of NN are")
    qualityofclassifier(NN_Pred,y_test)

    #When performance is observed for all the classifiers with varied parameters tuned, it was observed that Neural Network showed to be more accurate
    #than all the other classifiers. When tested with small datasets, Decision tree classifer showed more accuracy but when tested with whole dataset,
    #Decision tree accuracy decreased. While other parameter like F1 score is considered, Random Forest has more F1 score. Since the given dataset is
    #imbalanced, F1 score can be considered as an overall performance evaluator. So in this case, Random Forest performed best.
    #


def qualityofclassifier(prediction,testlabel):
    print("Confusion Matrix :" + str(confusion_matrix(prediction, testlabel)))
    print("Classification Report: "+ str(classification_report(prediction,testlabel)))
    print("Accuracy is: "+ str(accuracy_score(prediction,testlabel)))

    #It is important to calculate the performance of model when a classifier is applied on data and how the predictions are accurate to the original.
    #So different performance metrics such as Accuracy, Confusion Matrix,Precision, Recall, f1 score are calculated. Since Precision and Recall are 2 terms, a
    #single term F1 score which is harmonic mean of precision and recall. Since the data is imbalanced data, f1 score best describes the performance
    # of the classifier



FirstDF=dataextraction()  #FirstDF is Data Frame with headline, text, bip:topics, dc.date.published, itemid, XMLfilename
ExtractedFeatureDataframe=featureextraction(FirstDF) #Dataframe with faetures and labels
trainclassification(ExtractedFeatureDataframe) #Classifiers








