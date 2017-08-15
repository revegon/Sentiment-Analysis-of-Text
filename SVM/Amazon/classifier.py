from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
path = r'G:\Academic\ML\Sentiment-Analysis-of-Text\SVM\Amazon'
dataset = load_files(path, shuffle= False, decode_error='ignore', random_state=None,load_content=True)
trainData,testData,trainTarget,testTarget = train_test_split(dataset.data,dataset.target,train_size  = 0.88, test_size=0.12,random_state=45);
vectorizer=TfidfVectorizer(use_idf=True,lowercase = True, analyzer='word')
trainData=vectorizer.fit_transform(trainData)
print(trainData.shape)
from sklearn import svm
from sklearn.metrics import accuracy_score
# target_names = ['accident','art','crime','economics','education','entertainment','environment','international','opinion','politics','science_tech','sports']
model = svm.SVC(kernel='linear', C=1, gamma=1)
model.fit(trainData, trainTarget)
new_doc_tfidf_matrix = vectorizer.transform(testData)
predicted = model.predict(new_doc_tfidf_matrix)
print(accuracy_score(testTarget, predicted)) 
joblib.dump(vectorizer,"vectorizer.pkl")
joblib.dump(model,"trainer.pkl")
# print(metrics.classification_report(testTarget, predicted,target_names=target_names))
# print(confusion_matrix(predicted, testTarget, target_names))

