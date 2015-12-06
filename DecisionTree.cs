using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace DataMining_Assignment_5
{
    /// <summary>
    /// 决策树节点
    /// </summary>
    class DecisionTreeNode
    {
        
        /// <summary>
        /// 该节点祖先节点的划分合集
        /// </summary>
        public Split split;

        /// <summary>
        /// 检测标记，判断节点是否为叶子节点
        /// -1表示不是叶子节点
        /// 0,1表示叶子节点的类标
        /// </summary>
        public int checkFlag;

        public struct typeSplitCriteria
        {
            public int splitFeature;
            public double threshold;
        }
        /// <summary>
        /// 表示当前节点的划分标准;
        /// splitFeature 代表选取第i个特征
        /// 若为连续特征:threshold表示阈值
        /// </summary>
        public typeSplitCriteria curSplitCriteria;

        /// <summary>
        /// 子节点
        /// </summary>
        public DecisionTreeNode[] children;

        /// <summary>
        /// 当前节点根据出现次数确定的默认类标
        /// </summary>
        public int defaultLabel;

        /// <summary>
        /// 构造函数
        /// </summary>
        public DecisionTreeNode()
        {
            checkFlag = -1;
            children = null;
        }

    }
    /// <summary>
    /// 决策树(采用 C4.5)
    /// </summary>
    class DecisionTree
    {
        
        /// <summary>
        /// 根节点
        /// </summary>
        public DecisionTreeNode Root
        {
            get
            {
                return root;
            }
        }
        private DecisionTreeNode root;

        private DataSet dataset;

        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="ds">数据集</param>
        public DecisionTree(DataSet ds)
        {
            dataset = ds;
            root = new DecisionTreeNode();
            root.split = new Split(ds.FeatureType.Length);
        }

        /// <summary>
        /// 对数据集应用划分，返回符合划分的子集
        /// </summary>
        /// <param name="split">一个划分</param>
        /// <param name="raw">原始数据集</param>
        /// <returns>符合划分的子集</returns>
        public DataSet ApplySplit(Split split, DataSet raw)
        {
            DataSet result = new DataSet();
            result.FeatureType = raw.FeatureType;
            foreach (var d in raw.Examples)
            {
                if (split.Fit(d) == true) result.Examples.Add(d);
            }
            result.RecordDiscreteFeature();
            return result;
        }

        /// <summary>
        /// 判断当前点的数据集和划分是否符合一些基本情况
        /// 0:不符合任何基本情况; 
        /// 1,3:类标直接赋值为数据集中出现最多的类标; 
        /// 2:类标赋值为默认类标
        /// </summary>
        /// <param name="curSet">当前数据集</param>
        /// <param name="split">当前划分</param>
        /// <returns>0:不符合任何基本情况; 1,3:类标直接赋值为数据集中出现最多的类标; 2:类标赋值为默认类标</returns>
        private int CheckBaseCase(DataSet curSet, Split split)
        {
            if (curSet.Examples.Count == 0) return 2;//当前节点已经没有数据

            int i;
            for (i = 0; i < curSet.FeatureType.Length; i++)
            {
                if (split.splitOption[i] == 0) break;
            }
            if (i == curSet.FeatureType.Length) return 3;//不存在未使用的特征

            Example[] tmp = curSet.Examples.ToArray();
            int j;
            for (j = 1; j < tmp.Length; j++)
            {
                if (tmp[j].label != tmp[j - 1].label) break;
            }
            if (j == tmp.Length) return 1;//所有类标相同

            return 0;
        }

        #region 信息熵相关计算

        /// <summary>
        /// 计算当前数据集的信息熵
        /// </summary>
        /// <param name="curSet">数据集</param>
        /// <returns>信息熵</returns>
        private double Entropy(DataSet curSet)
        {
            double result = 0;
            double[] cntLabel = new double[2];
            cntLabel.Initialize();
            foreach (var d in curSet.Examples) cntLabel[d.label] += d.weight;
            int n = curSet.Examples.Count;
            for (int i = 0; i < 2; i++)
            {
                double p = (double)cntLabel[i] / (double)n;
                result += -1 * (p * Math.Log(p, 2));
            }
            return result;
        }

        /// <summary>
        /// 当前划分的信息熵（离散特征）
        /// </summary>
        /// <param name="curSet">数据集</param>
        /// <param name="splitFeature">划分特征</param>
        /// <param name="metric">当前划分的信息度量，用来归一化信息增益</param>
        /// <returns>划分的信息熵</returns>
        private double SplitEntropy_Discrete(DataSet curSet, int splitFeature, out double metric)
        {
            double result = 0;
            metric = 0;
            int N = curSet.Examples.Count;
            foreach(var f in curSet.DiscreteFeature[splitFeature])
            {
                //枚举当前特征所有可能取值
                //计算splitFeature在当前取值下的信息熵
                double[] cntLabel = new double[2];
                cntLabel.Initialize();
                foreach (var d in curSet.Examples)
                {
                    if (d.features[splitFeature] == f) cntLabel[d.label] += d.weight;
                }
                double n = cntLabel[0] + cntLabel[1];
                if (n == 0) continue;
                double tmp = 0;
                for (int i = 0; i < 2; i++)
                {
                    double p = (double)cntLabel[i] / (double)n;
                    if (p == 0) continue;
                    tmp += ((double)n / (double)N) * (-1 * (p * Math.Log(p, 2)));
                }
                result += tmp;

                //计算信息度量
                double pm = (double)n / (double)N;
                metric += pm * Math.Log(pm, 2);
            }
            metric *= -1;
            return result;
        }

        /// <summary>
        /// 返回所有可能的阈值
        /// </summary>
        /// <param name="curSet">数据集</param>
        /// <param name="splitFeature">划分特征</param>
        /// <returns>可能的阈值集合</returns>
        private double[] PossibleThreshold(DataSet curSet, int splitFeature)
        {
            double[] arr = new double[curSet.Examples.Count];
            int cnt = 0;
            foreach(var d in curSet.Examples)
            {
                arr[cnt] = d.features[splitFeature];
                cnt++;
            }
            /*
            使用所有取值的平均数作为阈值
            double[] result = new double[1];
            result[0] = arr.Average();
            return result;
            */
            return arr;//使用所有取值作为阈值的枚举
        }

        /// <summary>
        /// 当前划分的信息熵（连续特征）
        /// </summary>
        /// <param name="curSet">数据集</param>
        /// <param name="splitFeature">划分特征</param>
        /// <param name="bestThreshold">阈值</param>
        /// <param name="metric">当前划分的信息度量，用来归一化信息增益</param>
        /// <returns>划分的信息熵</returns>
        private double SplitEntropy_Numeric(DataSet curSet, int splitFeature, out double bestThreshold, out double metric)
        {
            bestThreshold = 0;
            double result = Double.MaxValue;
            metric = 0;
            int N = curSet.Examples.Count;
            foreach (var threshold in PossibleThreshold(curSet,splitFeature))
            {
                //枚举所有可能的阈值
                double tmpResult = 0;
                double[] cntLabel_less = new double[2];//小于等于阈值的数据
                cntLabel_less.Initialize();
                double[] cntLabel_more = new double[2];//大于阈值的数据
                cntLabel_more.Initialize();
                foreach(var ex in curSet.Examples)
                {
                    if (ex.features[splitFeature]<=threshold)
                    {
                        //小于等于阈值
                        cntLabel_less[ex.label] += ex.weight;
                    }
                    else
                    {
                        //大于阈值
                        cntLabel_more[ex.label] += ex.weight;
                    }
                }
                //小于等于阈值
                double n_less = cntLabel_less[0] + cntLabel_less[1];
                double tmp = 0;
                for (int i = 0; i < 2; i++)
                {
                    if (n_less == 0) continue;
                    double p = (double)cntLabel_less[i] / (double)n_less;
                    if (p == 0) continue;
                    tmp += ((double)n_less / (double)N) * (-1 * (p * Math.Log(p, 2)));
                }
                tmpResult += tmp;
                
                //大于阈值
                double n_more = cntLabel_more[0] + cntLabel_more[1];
                tmp = 0;
                for (int i = 0; i < 2; i++)
                {
                    if (n_more == 0) continue;
                    double p = (double)cntLabel_more[i] / (double)n_more;
                    if (p == 0) continue;
                    tmp += ((double)n_more / (double)N) * (-1 * (p * Math.Log(p, 2)));
                }
                tmpResult += tmp;

                if (tmpResult<result)
                {
                    //选择使用当前特征划分时，信息熵最小，即信息熵增益最大的阈值
                    result = tmpResult;
                    bestThreshold = threshold;
                    metric = 0;
                    double pm = (double)n_less / (double)N;
                    if (pm != 0) metric += pm * Math.Log(pm, 2);
                    pm = (double)n_more / (double)N;
                    if (pm != 0) metric += pm * Math.Log(pm, 2);
                }
            }
            metric *= -1;
            return result;
        }


        #endregion

        /// <summary>
        /// 对cur节点进行划分
        /// </summary>
        /// <param name="cur">要划分的节点</param>
        /// <param name="defaultLabel">默认类标，由父节点出现次数最多的类标决定</param>
        public void GenerateSplit(DecisionTreeNode cur, int defaultLabel)
        {
            //PrintLog(depth.ToString());
            DataSet curSet = ApplySplit(cur.split, dataset);
            //计算当前节点出现次数最多的类标，作为子节点的默认类标
            int newDefaultLabel;
            int[] cntLabel = new int[2];
            cntLabel[0] = cntLabel[1] = 0;
            foreach (var d in curSet.Examples) cntLabel[d.label]++;
            if (cntLabel[0] > cntLabel[1]) newDefaultLabel = 0;
            else newDefaultLabel = 1;
            cur.defaultLabel = newDefaultLabel;
            //检查是否符合几种基本情况
            int flag = CheckBaseCase(curSet, cur.split);
            if (flag != 0)
            {
                //符合基本情况
                if (flag == 2)
                {
                    //当前类标赋值为父类出现次数最多的类标
                    cur.checkFlag = defaultLabel;
                }
                if (flag == 1 || flag == 3)
                {
                    //寻找数据集中出现次数最多的类标
                    cur.checkFlag = newDefaultLabel;
                }
            }
            else
            {
                //需要选择一个未划分的特征进行划分
                double E_raw = Entropy(curSet);//当前集合的信息熵
                int bestFeature = 0;//当前最优划分特征
                double bestThreshold = 0;//最优划分特征的阈值（对连续特征而言）
                double maxIGR = 0;//最大的信息增益率
                //寻找信息熵增益率最大的未使用特征
                for (int i = 0; i < curSet.FeatureType.Length; i++)
                {
                    if (cur.split.splitOption[i]==0)
                    {
                        //未使用特征
                        if (curSet.FeatureType[i]==1)
                        {
                            //离散特征
                            double H;
                            double E_split = SplitEntropy_Discrete(curSet, i,out H);
                            double tmp = (E_raw - E_split) / H;
                            if (tmp>maxIGR)
                            {
                                maxIGR = tmp;
                                bestFeature = i;
                            }
                        }
                        else
                        {
                            //连续特征
                            double threshold;
                            double H;
                            double E_split = SplitEntropy_Numeric(curSet, i,out threshold, out H);
                            double tmp = (E_raw - E_split) / H;
                            if (tmp > maxIGR)
                            {
                                maxIGR = tmp;
                                bestFeature = i;
                                bestThreshold = threshold;
                            }
                        }
                    }
                }
                if (maxIGR == 0)
                {
                    //信息增益为0，则不划分
                    cur.checkFlag = curSet.Examples[0].label;
                    return;
                }
                //对最优的特征进行划分，创建子树
                //记录当前节点的划分标准
                cur.curSplitCriteria = new DecisionTreeNode.typeSplitCriteria();
                cur.curSplitCriteria.splitFeature = bestFeature;
                //创建子树
                if (curSet.FeatureType[bestFeature] == 0)
                {
                    //连续特征
                    //创建小于等于阈值的左子树 和 大于阈值的右子树
                    cur.curSplitCriteria.threshold = bestThreshold;
                    DecisionTreeNode lessNode = new DecisionTreeNode();
                    lessNode.split = new Split(cur.split);
                    lessNode.split.splitCriteria[bestFeature]=bestThreshold;
                    lessNode.split.splitOption[bestFeature] = -1;
                    DecisionTreeNode moreNode = new DecisionTreeNode();
                    moreNode.split = new Split(cur.split);
                    moreNode.split.splitCriteria[bestFeature] = bestThreshold;
                    moreNode.split.splitOption[bestFeature] = 1;
                    cur.children = new DecisionTreeNode[2];
                    cur.children[0] = lessNode;
                    cur.children[1] = moreNode;
                    //继续扩展子节点
                    GenerateSplit(cur.children[0], newDefaultLabel);
                    GenerateSplit(cur.children[1], newDefaultLabel);

                }
                else
                {
                    //离散特征
                    cur.children = new DecisionTreeNode[curSet.DiscreteFeature[bestFeature].Count()];
                    int cntChildren = 0;
                    foreach(var num in curSet.DiscreteFeature[bestFeature])
                    {
                        //对每个离散特征的值创建节点
                        DecisionTreeNode newNode = new DecisionTreeNode();
                        newNode.split = new Split(cur.split);
                        newNode.split.splitCriteria[bestFeature] = num;
                        newNode.split.splitOption[bestFeature] = 2;
                        cur.children[cntChildren] = newNode;
                        cntChildren++;
                    }
                    foreach(var node in cur.children)
                    {
                        //继续扩展子节点
                        GenerateSplit(node, newDefaultLabel);
                    }
                }
            }
        }

        /// <summary>
        /// 对data做测试
        /// </summary>
        /// <param name="data">测试数据</param>
        /// <returns>类标</returns>
        public int Test(Example data)
        {
            DecisionTreeNode cur = root;
            int defaultLabel=root.defaultLabel;
            while (cur!=null && cur.checkFlag == -1)
            {
                defaultLabel = cur.defaultLabel;
                int sf = cur.curSplitCriteria.splitFeature;
                if (dataset.FeatureType[sf]==0)
                {
                    //连续
                    double thres = cur.curSplitCriteria.threshold;
                    if (data.features[sf] <= thres) cur = cur.children[0];
                    else cur = cur.children[1];
                }
                else
                {
                    //离散
                    DecisionTreeNode next = null;
                    foreach(var child in cur.children)
                    {
                        if (child.split.splitCriteria[sf]==data.features[sf])
                        {
                            next = child;
                            break;
                        }
                    }
                    cur = next;
                }
            }
            if (cur!=null && cur.checkFlag != -1) return cur.checkFlag;
            else return defaultLabel;
        }
    }
}
