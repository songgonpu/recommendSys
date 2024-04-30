import pandas as pd
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import json
import pickle

def writeJsonFile(dictData, json_file_path):
    """
    写json 变量数据到文件中
    :param dictData:
    :param json_file_path:
    :return:
    """
    with open(json_file_path, 'w') as file:
        json.dump(dictData, file)


def readJsonFile(json_file_path):
    """
    从文件中读取json串出来
    :param json_file_path:
    :return:
    """
    with open(json_file_path, 'r') as file:
        loaded_dict = json.load(file)
    return loaded_dict


def writePickleFile(obj, filePath):
    """
    将python对象保存文件
    :param obj:
    :param filePath:
    :return:
    """
    with open(filePath, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def readPickleFile(filePath):
    """
    读取Pickle文件并返回
    :param filePath:
    :return:
    """
    with open(filePath, 'rb') as handle:
        loaded_obj = pickle.load(handle)

    return loaded_obj

class DialogRecommendModelV2:
    """
    对话推荐模型

    包含数据预处理，训练、推荐
    ...

    author : songgongpu
    date   : 2024-04-26
    """

    def __init__(self, mappingFilePath, itemAfterFeaturePath, allFeatureListPath, model_path=None):
        """
        数据初始化，将文件数据读取出来合并到一起
        :param mappingFilePath:
        :param itemAfterFeaturePath:
        :param allFeatureListPath:
        :param model_path:
        """

        self.interactions = None
        self.itemDF = None
        self.userDF = None

        # 模型
        self.model = None
        # 所有的特征（features）
        self.feature_columns = None
        # 模型保存的路径
        self.mode_path = model_path
        # mapping文件保存地址
        self.mappingFilePath = mappingFilePath
        # 特征化后的物品文件保存地址
        self.itemAfterFeaturePath = itemAfterFeaturePath
        # 全特征embedding信息保存地址
        self.allFeatureListPath = allFeatureListPath

        # 离散标签映射
        self.labelMapping = {}


    def loadLocalFile(self, userDataPath, itemDataPath, historyDataPath):
        """
        加载用户、物品、行为 三种数据
        :param userDataPath:
        :param itemDataPath:
        :param historyDataPath:
        :return:
        """

        # 用户数据
        userDF = pd.read_csv(userDataPath, names=['userId', 'psncode', 'psnname', 'deptname', 'jobtitle', 'sex', 'country', 'companycode'], skiprows=1)
        # 物品数据
        itemDF = pd.read_csv(itemDataPath, names=['itemId', 'item_name', 'item_type', 'category'], skiprows=1)
        # 行为数据
        historyDF = pd.read_csv(historyDataPath, names=['userId', 'itemId', 'numb'], skiprows=1)

        # 将用户特征、物品特征合并到交叉数据中
        interactions = pd.merge(historyDF, userDF, on='userId', how='left')
        interactions = pd.merge(interactions, itemDF, on='itemId', how='left')

        # 赋值
        self.interactions = interactions
        # 转换之后的ItemDF
        self.itemDF = itemDF
        # 转换之后的UserDF
        self.userDF = userDF


    def featurizeData(self, userSpareFeatureList, itemSpareFeatureList, denseFeatureList):
        """
        特征化处理，将原始数据特征化处理。
        分别进行，离散归一处理
        :param userSpareFeatureList:
        :param itemSpareFeatureList:
        :param denseFeatureList:
        :return:
        """

        lbe = LabelEncoder()

        # 离散的用户特征
        for feat in userSpareFeatureList:
            tempList = lbe.fit_transform(self.userDF[feat])

            # 当前类型不在labelObj内，则写入Mapping
            if not self.labelMapping or feat not in self.labelMapping:
                temp_dict = dict(zip(self.userDF[feat], tempList))
                # 写入mapping中
                self.labelMapping[feat] = temp_dict
            self.userDF[feat] = tempList

        # 离散物品特征
        for feat in itemSpareFeatureList:
            tempList = lbe.fit_transform(self.itemDF[feat])

            # 当前类型不在labelObj内，则写入Mapping
            if not self.labelMapping or feat not in self.labelMapping:
                temp_dict = dict(zip(self.itemDF[feat], tempList))
                # 写入mapping中
                self.labelMapping[feat] = temp_dict
            self.itemDF[feat] = tempList

        # 更新行为中的用户、物品特征
        for feat in list(set(userSpareFeatureList + itemSpareFeatureList)):
            if feat in self.labelMapping.keys():
                # 给物品数据集映射特征
                tempList = [self.labelMapping[feat][ele] for ele in self.interactions[feat]]
                self.interactions[feat] = tempList

        # 连续特征处理
        mms = MinMaxScaler()
        self.interactions[denseFeatureList] = mms.fit_transform(self.interactions[denseFeatureList])

        # DataFrame 保存到文件
        writePickleFile(self.itemDF, self.itemAfterFeaturePath)
        # 写入离散特征映射文件中
        writePickleFile(self.labelMapping, self.mappingFilePath)

        return self.interactions


    def featureCategoryInfo(self, interactions, userSpareFeatureList, itemSpareFeatureList, denseFeatureList):
        """
        对特征结构进行处理，初始化模型用
        :param userSpareFeatureList:
        :param itemSpareFeatureList:
        :param denseFeatureList:
        :return:
        """

        full_spase_features_list = list(set(userSpareFeatureList + itemSpareFeatureList))
        full_spase_features_list.append("userId")
        full_spase_features_list.append("itemId")

        # 定义特征列
        sparse_user_features = [SparseFeat(feat, vocabulary_size=interactions[feat].nunique(), embedding_dim=4) for feat in full_spase_features_list]
        dense_user_features = [DenseFeat(feat, 1) for feat in denseFeatureList]
        # 全特征
        self.feature_columns = sparse_user_features + dense_user_features

        # 保存到文件
        writePickleFile(self.feature_columns, self.allFeatureListPath)


    def build_input_model_data(self, dataFrame):
        """
        基于给到的数据集，转换为模型训练 或 预测的格式
        :param dataList:
        :return:
        """
        return {name: dataFrame[name].values for name in get_feature_names(self.feature_columns)}

    def train(self, userDataPath, itemDataPath, historyDataPath, userSpareFeatureList, itemSpareFeatureList, denseFeatureList, target_column, **kwargs):
        """

        :param sparse_features_list:
        :param dense_features_list:
        :param target_column:
        :param kwargs:
        :return:
        """

        # 1， 加载原始数据
        self.loadLocalFile(userDataPath, itemDataPath, historyDataPath)

        # 2， 原始数据特征化处理，并返回处理后的全量数据
        interactions = self.featurizeData(userSpareFeatureList, itemSpareFeatureList, denseFeatureList)

        # 3， 生成特征处理embedding规则
        self.featureCategoryInfo(interactions, userSpareFeatureList, itemSpareFeatureList, denseFeatureList)

        # 4， 分隔数据，训练集与测试集
        train, test = train_test_split(interactions, test_size=0.1, random_state=42)

        # 5， 构建训练与测试输入数据集
        train_input = self.build_input_model_data(train)
        test_input = self.build_input_model_data(test)

        print("train_input is : {}".format(train_input))

        # 6， 初始化DeepFM模型、编译
        self.model = DeepFM(linear_feature_columns=self.feature_columns, dnn_feature_columns=self.feature_columns, task='binary') # 目标为二分类
        self.model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'])

        # 7， 训练模型
        history = self.model.fit(train_input, train[target_column].values, batch_size=64, epochs=5, verbose=2, validation_split=0.2)
        print("history is :{}".format(history))

        # 8， 保存模型文件
        if self.mode_path:
            self.model.save(self.mode_path)

        # 9， 评估模型
        result = self.model.evaluate(test_input, test[target_column].values, batch_size=64)
        print("Test Loss and Accuracy:", result)

    def prePredictData(self, userDict, userSpareFeatureList):
        """
        将要预测的用户映射为规则化的数据，并与物品列表同时合并
        :param userDict:
        :return:
        """
        userDF = pd.DataFrame(userDict)
        for feat in userSpareFeatureList:
            if feat in self.labelMapping.keys():
                # 给物品数据集映射特征
                userDF[feat] = [self.labelMapping[feat].get(ele, None) for ele in userDF[feat]]

        # 用户与物品合并
        predictList = pd.merge(userDF, self.itemDF, how="cross").drop_duplicates()
        predictList['numb'] = 0

        print("predictList top 5 is : {}".format(predictList.head(5)))

        return predictList

    def predict(self, user_dict, userSpareFeatureList, top_n=5):
        if not self.model:
            if self.mode_path:
                # 模型不存在，模型路径存在，加载模型、加载物品列表
                self.model = tf.keras.models.load_model(self.mode_path)
                self.itemDF = readPickleFile(self.itemAfterFeaturePath)
                self.feature_columns = readPickleFile(self.allFeatureListPath)
                self.labelMapping = readPickleFile(self.mappingFilePath)

            else:
                raise ValueError("Model has not been trained or loaded.")

        predictList = self.prePredictData(user_dict, userSpareFeatureList)
        input_predict = self.build_input_model_data(predictList)

        print("input_predict is : {}".format(input_predict))

        score = self.model.predict(input_predict)

        from itertools import chain
        flat_list = list(chain(*score))
        # print(flat_list)

        newItemDF = self.itemDF
        newItemDF['score'] = pd.Series(flat_list)
        newItemDF['newItem'] = [model.find_key_by_value(model.labelMapping['item_name'], ele) for ele in newItemDF['item_name']]
        newItemDF['newItemType'] = [model.find_key_by_value(model.labelMapping['item_type'], ele) for ele in newItemDF['item_type']]
        newItemDF['newCategory'] = [model.find_key_by_value(model.labelMapping['category'], ele) for ele in newItemDF['category']]

        print("newItemDF is :{}".format(newItemDF.head(5)))
        resultDF = newItemDF[['newItem', 'newItemType', 'newCategory', 'score']]

        sorted_recommendations = resultDF.sort_values(by='score', ascending=False).head(top_n)

        print("输出预测结果....")
        print("type(score) is : {}".format(type(sorted_recommendations)))
        print("score is : {}".format(sorted_recommendations))
        return sorted_recommendations

    def find_key_by_value(self, input_dict, value_to_find):
        """
        反向查 Key
        :param input_dict:
        :param value_to_find:
        :return:
        """
        for key, value in input_dict.items():
            if value == value_to_find:
                return key
        return None


if __name__ == "__main__":

    # 用户离散特征列表
    userSpareFeatureList = ["deptname", "jobtitle", "sex", "country", "companycode"]
    # 物品离散特征列表
    itemSpareFeatureList = ["item_name", "item_type", "category"]
    # 连续特征列表
    denseFeatureList = ["numb"]

    # 源数据-用户
    userDataPath = "./data/recommend_source_staff.csv"
    # 源数据-物品
    itemDataPath = "./data/recommend_source_item.csv"
    # 源数据-历史
    historyDataPath = "./data/recommend_source_history.csv"

    # 特征映射文件保存路径
    mappingFilePath = "./model/mappingFile"
    # 物品特征数据文件保存路径
    itemAfterFeaturePath = "./model/itemFeature"
    # 全特征embedding保存对象
    allFeatureListPath = "./model/featureColumn"

    # 初始化
    model = DialogRecommendModelV2(mappingFilePath, itemAfterFeaturePath, allFeatureListPath, model_path="./model/dialogRecommend0.1")

    # 训练
    target_column = "numb"
    model.train(userDataPath, itemDataPath, historyDataPath, userSpareFeatureList, itemSpareFeatureList, denseFeatureList, target_column)

    print("self.labelMapping is {}".format(model.labelMapping))
    print("self.labelMapping.companycode 210991 is : {}".format(model.labelMapping['companycode'].get(210991)))

    # 预测
    user_predict = {'userId':[97159], 'deptname':['人工智能组'], 'jobtitle':['大数据开发工程师'], 'sex':['男'], 'country':['CHN'], 'companycode':[210991]}
    recommendations = model.predict(user_dict=user_predict, userSpareFeatureList=userSpareFeatureList, top_n=10)

    # 打印结果
    print("result is : {}".format(recommendations))
    print("ending...")
