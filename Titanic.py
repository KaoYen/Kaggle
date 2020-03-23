import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


def data_preprocess():
    """ 觀察資料集是否有缺失值"""
    train.info()  # had loss value: Age、Cabin、Embarked
    test.info()  # had loss value: Age、Fare、Cabin

    # 使用中位數來填補缺失值
    train["Age"] = train["Age"].fillna(train["Age"].median())
    test["Age"] = test["Age"].fillna(test["Age"].median())
    train.info()
    test.info()

    # male男性轉為 0 、 female女性轉為 1
    train.loc[train["Sex"] == "male", "Sex"] = 0
    train.loc[train["Sex"] == "female", "Sex"] = 1
    test.loc[test["Sex"] == "male", "Sex"] = 0
    test.loc[test["Sex"] == "female", "Sex"] = 1

    # 使用中位數來填補缺失值
    train["Fare"] = train["Fare"].fillna(train['Fare'].median())
    test["Fare"] = test["Fare"].fillna(test['Fare'].median())

    # 將缺失欄位都設為S港口 & value 轉成int
    train["Embarked"] = train["Embarked"].fillna("S")
    train.loc[train["Embarked"] == "S", "Embarked"] = 0
    train.loc[train["Embarked"] == "C", "Embarked"] = 1
    train.loc[train["Embarked"] == "Q", "Embarked"] = 2
    test["Embarked"] = test["Embarked"].fillna("S")
    test.loc[test["Embarked"] == "S", "Embarked"] = 0
    test.loc[test["Embarked"] == "C", "Embarked"] = 1
    test.loc[test["Embarked"] == "Q", "Embarked"] = 2


def random_forest():
    data_preprocess()

    # sns.countplot(train['Sex'], hue=train['Survived'])  # Sex
    # plt.show()
    # display(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().round(3))
    #
    # sns.countplot(train['Pclass'], hue=train['Survived'])  # Pclass
    # display(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().round(3))
    # plt.show()

    # input set and labels
    # X = train.drop(labels=['Survived', 'PassengerId'], axis=1)
    # Y = train['Survived']
    #
    # Base = ['Sex', 'Pclass']
    # Base_Model = RandomForestClassifier(random_state=2, n_estimators=300, min_samples_split=20, oob_score=True)
    # Base_Model.fit(X[Base], Y)
    # print(f'Base oob score :{Base_Model.oob_score_:.3f}')

    """train model"""
    predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked"]  # 預測特徵宣告
    RFC = RandomForestClassifier(random_state=2, n_estimators=100, min_samples_split=20, oob_score=True)
    RFC.fit(train[predictors], train["Survived"])
    print(RFC.oob_score_)

    """predict test dataset"""
    pred = RFC.predict(test[predictors])
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred
    })
    submission.to_csv('submission.csv', index=False)


def decision_tree():
    data_preprocess()
    predictors = ["Pclass", "Sex", "Age"]  # 預測特徵宣告
    clf = tree.DecisionTreeClassifier()
    clf.fit(train[predictors], train["Survived"])
    pred = clf.predict(test[predictors])

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred
    })
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    # random_forest()
    decision_tree()
