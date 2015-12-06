using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataMining_Assignment_5
{
    class Pair
    {
        public DecisionTree tree;
        public double coefficient;
    }
    class AdaBoost
    {
        private List<Pair> weakTrees;
        private DataSet dataset;
        public AdaBoost(DataSet ds)
        {
            dataset = ds;
            foreach (var ex in dataset.Examples) ex.weight = 1;
            weakTrees = new List<Pair>();
        }

        /// <summary>
        /// 执行带Adaboost的决策树生成
        /// </summary>
        /// <param name="T">迭代次数</param>
        public void Generate(int T)
        {
            for (int t = 0; t < T; t++)
            {
                DecisionTree tree = new DecisionTree(dataset);
                tree.GenerateSplit(tree.Root, 0);
                int n = dataset.Examples.Count;
                double eps = 0;
                foreach (var ex in dataset.Examples)
                {
                    if (tree.Test(ex) != ex.label) eps += ex.weight / n;
                }
                //计算系数
                double alpha = 0.5 * Math.Log((1 - eps) / eps);
                //记录该次迭代产生的决策树和系数
                Pair pair = new Pair();
                pair.tree = tree;
                pair.coefficient = alpha;
                weakTrees.Add(pair);
                //更新数据权值
                double sum = 0;
                foreach (var ex in dataset.Examples)
                {
                    //w=w*exp(-y*Test(x)*alpha), 类标为-1,1
                    if (tree.Test(ex) == ex.label)
                    {
                        ex.weight *= Math.Exp(-1 * alpha);
                    }
                    else
                    {
                        ex.weight *= Math.Exp(alpha);
                    }
                    sum += ex.weight;
                }
                //归一化权重（为了方便决策树计算，归一化后乘以n）
                foreach (var ex in dataset.Examples)
                {
                    ex.weight = ex.weight / sum * n;
                }
                double tmp = 0;
                foreach (var ex in dataset.Examples)
                {
                    tmp += ex.weight;
                }
                if (alpha < 0.00001) break;
            }
        }

        /// <summary>
        /// 对数据data做测试
        /// </summary>
        /// <param name="data"></param>
        /// <returns>类标</returns>
        public int Test(Example data)
        {
            double result = 0;
            foreach(var pair in weakTrees)
            {
                int testLabel = pair.tree.Test(data);
                if (testLabel == 0) result += -1 * pair.coefficient;
                else result += pair.coefficient;
            }
            if (result > 0) return 1;
            else return 0;
        }
            
    }
}
