using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataMining_Assignment_5
{
    class RandomForest
    {
        private List<DecisionTree> forest;
        private DataSet dataset;
        private Random rand;
        public RandomForest(DataSet ds)
        {
            dataset = ds;
            forest = new List<DecisionTree>();
            rand = new Random();
        }

        /// <summary>
        /// 生成随机森林，共n棵决策树，每次采样k个特征
        /// </summary>
        /// <param name="n">决策树的个数</param>
        /// <param name="k">每次采样的特征数</param>
        public void GenerateForest(int n, int k)
        {
            for (int i = 0; i < n; i++)
            {
                DataSet newSet = new DataSet();
                newSet.FeatureType = dataset.FeatureType;
                //对训练集的行进行随机采样
                int N = dataset.Examples.Count;
                var arrEx = dataset.Examples.ToArray();
                for (int j = 0; j < N; j++)
                {
                    //有放回随机
                    newSet.AddExample(arrEx[rand.Next() % N]);
                }
                newSet.RecordDiscreteFeature();
                DecisionTree tree = new DecisionTree(newSet);

                //对训练集的列进行随机采样（特征采样）
                int cntFeature = dataset.FeatureType.Length;
                Split newSplit = new Split(cntFeature);
                List<int> check = new List<int>();
                //随机选取cntFeature - k个特征去除
                for (int j = 0; j < cntFeature - k; j++)
                {
                    int f = rand.Next() % cntFeature;
                    while (check.Contains(f)==true) f = rand.Next() % cntFeature;//无放回随机
                    check.Add(f);
                    newSplit.splitOption[f] = 3;
                }
                tree.Root.split = new Split(newSplit);
                tree.GenerateSplit(tree.Root, 0);
                forest.Add(tree);
            }
        }

        /// <summary>
        /// 对data做测试
        /// </summary>
        /// <param name="data">测试数据</param>
        /// <returns>可能性最大的类</returns>
        public int Test(Example data)
        {
            int[] label = new int[2];
            label.Initialize();
            foreach (var tree in forest)
            {
                label[tree.Test(data)]++;
            }
            if (label[0] > label[1]) return 0;
            else return 1;
        }

    }
}
