using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataMining_Assignment_5
{
    /// <summary>
    /// 表示一条数据
    /// </summary>
    class Example
    {
        /// <summary>
        /// 特征
        /// </summary>
        public double[] features;
        /// <summary>
        /// 类标
        /// </summary>
        public int label;
        /// <summary>
        /// 权重, 用于Adaboost
        /// </summary>
        public double weight;
    }
}
