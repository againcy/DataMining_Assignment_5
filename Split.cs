using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataMining_Assignment_5
{
    class Split
    {
        /// <summary>
        /// 节点划分标准
        /// 对连续特征表示划分阈值
        /// 对离散特征表示特征取值
        /// </summary>
        public double[] splitCriteria;

        /// <summary>
        /// 划分选项
        /// 0表示未使用特征
        /// 2表示离散特征取值
        /// -1表示连续特征小于等于阈值
        /// 1表示连续特征大于阈值
        /// 3表示不使用该特征（随机森林）
        /// </summary>
        public int[] splitOption;

        public Split(int n)
        {
            splitCriteria = new double[n];
            splitOption = new int[n];
            splitOption.Initialize();
        }
        public Split(Split s)
        {
            int n = s.splitCriteria.Length;
            splitCriteria = new double[n];
            s.splitCriteria.CopyTo(splitCriteria,0);
            splitOption = new int[n];
            s.splitOption.CopyTo(splitOption, 0);
        }

        /// <summary>
        /// 判断给定的数据是否符合当前划分
        /// </summary>
        /// <param name="ex">一个数据</param>
        /// <returns>true:数据符合划分; false:数据不符合划分</returns>
        public bool Fit(Example ex)
        {
            for (int i = 0; i < ex.features.Length; i++)
            {
                if (splitOption[i] == 0 || splitOption[i] == 3)
                {
                    continue;//未使用该特征或不使用该特征
                }
                else if (splitOption[i] == 2)
                {
                    //离散数据
                    if (ex.features[i] != splitCriteria[i]) return false;
                }
                else if (splitOption[i] == 1 || splitOption[i] == -1)
                {
                    //连续数据
                    if (splitOption[i] == -1 && ex.features[i] > splitCriteria[i]) return false;
                    if (splitOption[i] == 1 && ex.features[i] <= splitCriteria[i]) return false;
                }
            }
            return true;
        }
    }
}
