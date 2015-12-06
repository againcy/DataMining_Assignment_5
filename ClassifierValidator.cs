using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataMining_Assignment_5
{
    class ValidationIndex
    {
        /// <summary>
        /// 准确度平均值
        /// </summary>
        public double Mean_accuracy
        {
            get
            {
                if (accuracy.Count == 0) return 0;
                else return accuracy.Average();
            }
        }
        /// <summary>
        /// 准确度标准差
        /// </summary>
        public double SD_accuracy
        {
            get
            {
                if (accuracy.Count == 0) return 0;
                else
                {
                    double mean = this.Mean_accuracy;
                    double tmp = 0;
                    foreach (var num in accuracy)
                    {
                        tmp += Math.Pow((num - mean), 2);
                    }
                    tmp /= accuracy.Count;
                    return Math.Sqrt(tmp);
                }
            }
        }

        private LinkedList<double> accuracy;
        public ValidationIndex()
        {
            accuracy = new LinkedList<double>();
        }
        /// <summary>
        /// 添加一次测试的准确度
        /// </summary>
        /// <param name="acc"></param>
        public void Add(double acc)
        {
            accuracy.AddLast(acc);
        }
    }
    class ClassifierValidator
    {
        PrintLog printLog;
        private delegate int TestFunction(Example ex);
        DataSet dataset;
        Random rand;
        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="ds">数据集</param>
        /// <param name="func">日志打印函数</param>
        public ClassifierValidator(DataSet ds, PrintLog func)
        {
            dataset = ds;
            rand = new Random();
            printLog = func;
        }
        DataSet trainingSet;
        DataSet testSet;
        DataSet[] subsets;

        /// <summary>
        /// 将原始数据集等分成k个子集
        /// </summary>
        /// <param name="k">数据平均分为k份</param>
        private void GenerateSets(int k)
        {
            subsets = new DataSet[k];
            int n = dataset.Examples.Count;
            Example[] arr = dataset.Examples.ToArray();
            //将原始数据集顺序打乱，则顺序取出的数据视作随机取出
            for(int i=0;i<arr.Length;i++)
            {
                int i1 = rand.Next() % n;
                int i2 = rand.Next() % n;
                var tmp = arr[i1];
                arr[i1] = arr[i2];
                arr[i2] = tmp;
            }
            //将原始数据集等分成k个子集
            int subLength = n / k;
            int cnt = 0;
            for (int i = 0; i < k; i++)
            {
                subsets[i] = new DataSet();
                subsets[i].FeatureType = dataset.FeatureType;
                for (int j = 0; j < subLength; j++)
                {
                    subsets[i].AddExample(arr[cnt]);
                    cnt++;
                    if (cnt >= arr.Length) break;
                }
            }
        }

        /// <summary>
        /// 选择测试集，并生成训练集
        /// </summary>
        /// <param name="id">测试集在subsets中的下标</param>
        /// <param name="k">子集的个数</param>
        private void ChooseTestSet(int id, int k)
        {
            testSet = subsets[id];
            trainingSet = new DataSet();
            trainingSet.FeatureType = testSet.FeatureType;
            for (int i = 0; i < k; i++)
            {
                if (i == id) continue;
                foreach (var ex in subsets[i].Examples) trainingSet.AddExample(ex);
            }
            trainingSet.RecordDiscreteFeature();
        }

        /// <summary>
        /// 根据相应的测试函数获取测试集的分类准确度
        /// </summary>
        /// <param name="test">测试函数</param>
        /// <returns>准确度</returns>
        private double GetAccuracy(TestFunction test)
        {
            //准确度
            int trueCount = 0;
            foreach (var ex in testSet.Examples)
            {
                int testLabel = test(ex);
                if (testLabel == ex.label) trueCount++;
            }
            double result = (double)trueCount / (double)testSet.Examples.Count;
            return result;
        }

        /// <summary>
        /// k折交叉评估
        /// 1:决策树;
        /// 2:随机森林
        /// 3:带Adaboost的决策树(T为迭代次数)
        /// </summary>
        /// <param name="k">数据平均分成的组数</param>
        /// <param name="mode">1:决策树; 2:随机森林; 3:带Adaboost的决策树</param>
        /// <param name="T">Adaboost的迭代次数</param>
        /// <returns></returns>
        public ValidationIndex CrossValidation(int k, int mode, int T = 30)
        {
            ValidationIndex result = new ValidationIndex();
            printLog("即将开始一次" + k.ToString() + "折交叉评估");
            GenerateSets(k);
            for (int repeat = 0; repeat < k; repeat++)
            {
                printLog("正在进行第" + (repeat + 1).ToString() + "折的分析...");
                ChooseTestSet(repeat, k);
                double curAcc = 0;
                switch (mode)
                {
                    case 1:
                        {
                            //普通决策树
                            DecisionTree classifier = new DecisionTree(trainingSet);
                            classifier.GenerateSplit(classifier.Root, 0);
                            curAcc = GetAccuracy(classifier.Test);
                            
                            break;
                        }
                    case 2:
                        {
                            //随机森林
                            RandomForest classifier = new RandomForest(trainingSet);
                            int cntEx = trainingSet.Examples.Count / 1;
                            int cntF = (int)Math.Sqrt((double)trainingSet.FeatureType.Length);
                            classifier.GenerateForest(cntEx, cntF);
                            curAcc = GetAccuracy(classifier.Test);
                            
                            break;
                        }
                    case 3:
                        {
                            //带Adaboost的决策树
                            AdaBoost classifier = new AdaBoost(trainingSet);
                            classifier.Generate(T);
                            curAcc = GetAccuracy(classifier.Test);
                            
                            break;
                        }
                };
                printLog("  该次准确率为 " + curAcc.ToString("0.00%"));
                result.Add(curAcc);
            }
            return result;
        } 
    }
}
