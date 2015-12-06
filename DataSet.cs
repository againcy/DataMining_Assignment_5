using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace DataMining_Assignment_5
{
    /// <summary>
    /// 数据集类
    /// </summary>
    class DataSet
    {
        /// <summary>
        /// 数据
        /// </summary>
        public List<Example> Examples
        {
            get
            {
                return examples;
            }
            set
            {
                examples = value;
            }
        }
        private List<Example> examples;
        /// <summary>
        /// 特征的类型1:离散, 0:连续
        /// </summary>
        public int[] FeatureType
        {
            get
            {
               return featureType;
            }
            set
            {
                featureType = value;
            }
        }
        private int[] featureType;

        /// <summary>
        /// 记录每个离散特征的取值
        /// </summary>
        public List<double>[] DiscreteFeature
        {
            get
            {
                return discreteFeature;
            }
            set
            {
                discreteFeature = value;
            }
        }
        private List<double>[] discreteFeature;

        public DataSet()
        {
            examples = new List<Example>();
        }

        /// <summary>
        /// 添加一个数据
        /// </summary>
        /// <param name="ex"></param>
        public void AddExample(Example ex)
        {
            examples.Add(ex);
        }

        /// <summary>
        /// 记录每个离散特征的取值
        /// </summary>
        public void RecordDiscreteFeature()
        {
            discreteFeature = new List<double>[featureType.Length];
            for (int i = 0; i < featureType.Length; i++)
            {
                if (featureType[i] == 1)
                {
                    discreteFeature[i] = new List<double>();
                    foreach (var d in examples)
                    {
                        if (discreteFeature[i].Contains(d.features[i]) == false) discreteFeature[i].Add(d.features[i]);
                    }
                }
            }
        }
        
        /// <summary>
        /// 读入数据集
        /// </summary>
        /// <param name="file">文件路径</param>
        public void Input(string file)
        {
            StreamReader sr = new StreamReader(file);
            
            string line = sr.ReadLine();
            featureType = line.Split(',').Select<string, int>(x => Convert.ToInt32(x)).ToArray();
            while ((line = sr.ReadLine()) != null)
            {
                Example tmp = new Example();
                double[] data = line.Split(',').Select<string, double>(x => Convert.ToDouble(x)).ToArray();
                tmp.features = data.Take(data.Length - 1).ToArray();
                //二元类，将-1转换为0，方便后续操作
                if (data[data.Length - 1] == 1) tmp.label = 1;
                else tmp.label = 0;
                tmp.weight = 1;
                examples.Add(tmp);
            }
            sr.Close();
            RecordDiscreteFeature();
        }
    }
}
