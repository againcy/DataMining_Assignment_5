# DataMining Assignment 5
基于c4.5决策树的RandomForest和Adaboost算法
## Example.cs
一条数据，存储特征的取值，类标以及权重
```c#
public double[] features;//特征
public int label;//类标
public double weight;//权重（用于Adaboost; 其他情况默认为1）
```
## DataSet.cs
一个数据集
```c#
private int[] featureType;//特征类型: 0表示连续数据; 1表示离散数据
private List<double>[] discreteFeature;//记录当前数据集离散特征的取值
public void AddExample(Example ex)//添加一条数据
public void RecordDiscreteFeature()//统计所有离散特征的取值
```
## Split.cs
用于记录决策树中每个节点的特征划分依据
```c#
public double[] splitCriteria;//对每个特征的划分标准: 连续数据表示特征划分阈值; 离散数据表示特征取值
public int[] splitOption;//节点的划分选项: 0表示未使用特征; 2表示离散特征; -1和1表示连续特征小于等于或大于阈值; 3表示不使用该特征（用于随机森林）
public bool Fit(Example ex)//判断一条数据是否符合当前划分标准
```
## DecisionTree.cs
```c#
class DecisionTreeNode
```
决策树的一个节点
```c#
public Split split;//当前节点的数据集满足的划分(由祖先决定)
public int checkFlag;//检测标记: -1节点非叶节点; 0,1:节点为叶节点, 同时表示该节点的类标
public typeSplitCriteria curSplitCriteria;//存储当前节点的划分标准, 用于构建子树: splitFeature代表选取第i个特征; threshold表示连续特征的阈值
public DecisionTreeNode[] children;//子节点
public int defaultLabel;//当前节点根据满足划分的数据集的最大可能类标决定的默认类标
```
```c#
class DecisionTree
```
C4.5决策树
```c#
private DataSet dataset;//数据集
private DecisionTreeNode root;//根节点
public DataSet ApplySplit(Split split, DataSet raw)//对数据集raw应用划分split，返回满足划分的所有数据组成的数据集
private int CheckBaseCase(DataSet curSet, Split split)//检查curSet是否满足基本情况(具体见代码内注释)
private double Entropy(DataSet curSet)//计算curSet的信息熵
//计算curSet在对splitFeature(离散)进行划分的信息熵, metric记录当前划分的信息度量由于归一化
private double SplitEntropy_Discrete(DataSet curSet, int splitFeature, out double metric)
private double[] PossibleThreshold(DataSet curSet, int splitFeature)//返回curSet的splitFeature进行划分时可能的阈值(通过枚举该特征的所有取值)
//计算curSet在对splitFeature(连续)进行划分的信息熵, metric记录当前划分的信息度量由于归一化, bestThreshold记录划分的阈值
private double SplitEntropy_Numeric(DataSet curSet, int splitFeature, out double bestThreshold, out double metric)
public void GenerateSplit(DecisionTreeNode cur, int defaultLabel)//对cur节点进行划分, defaultLabel为缺省类标, 由父节点的最大可能类标决定
public int Test(Example data)//返回data经过决策树测试后得到的类标
```
## RandomForest.cs
由```DecisionTree```构成的随机森林
```c#
private DataSet dataset;//数据集
private Random rand;//随机变量
private List<DecisionTree> forest;//随机森林
public void GenerateForest(int n, int k)//生成随机森林, 每次有放回随机选取n个数据, 同时随机选取k个特征进行决策树训练
public int Test(Example data)//对data进行测试, 返回可能性最大的类标
```
##AdaBoost.cs
```c#
class Pair
```
一个弱分类器
```c#
public DecisionTree tree;//一个决策树分类器
public double coefficient;//分类器的权重
```
```c#
class AdaBoost
```
AdaBoost算法生成的强分类器
* [参考资料](http://blog.csdn.net/haidao2009/article/details/7514787)
* 由于```DecisionTree```中计算信息熵时使用的权值和为n, 这里计算时先将权重weight=weight/n以归一化
```c#
private DataSet dataset;//数据集
private List<Pair> weakTrees;//弱分类器
public void Generate(int T)//生成强分类器, 一共迭代T次
public int Test(Example data)//对data进行测试，根据弱分类器的权重计算并返回最大可能的类标
```
##ClassifierValidator.cs
```c#
class ValidationIndex
```
k-折交叉评估的评估指标
```c#
public double Mean_accuracy//返回准确度平均值
public double SD_accuracy//返回准确度标准差
private LinkedList<double> accuracy;//各次测试的准确度
public void Add(double acc)//添加一次测试的准确度
```
```c#
class ClassifierValidator
```
分类器评估(k-折交叉评估)
```c#
DataSet dataset;//数据集
Random rand;//随机变量
DataSet trainingSet;//训练集
DataSet testSet;//测试集
DataSet[] subsets;k-折交叉评估中将数据随机分入的k个子集
private void GenerateSets(int k)//生成k个随机子集
private void ChooseTestSet(int id, int k)//选择subsets[id]作为测试集, 剩余k-1个子集合并为训练集
private double GetAccuracy(TestFunction test)//对测试函数Test获取测试集的准确度
//进行一次k-折交叉评估, mode表示分类器的类型:1 决策树; 2:随机森林; 3:AdaBoost, T表示AdaBoost的迭代次数(默认为30次)
public ValidationIndex CrossValidation(int k, int mode, int T = 30)
```
