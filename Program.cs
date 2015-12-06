using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace DataMining_Assignment_5
{
    delegate void PrintLog (string str);
    class Program
    {
        static private void PrintLog(string log)
        {
            Console.WriteLine(System.DateTime.Now.ToLongTimeString() + "  " + log);
        }
        static void CrossTest()
        {
            Console.WriteLine("输入需要使用的数据集文件名\n\t1: breast-cancer-assignment5.txt\n\t2: german-assignment5.txt\n\t其他请输入文件名");
            string x = Console.ReadLine();
            string file = "";

            if (x == "1")
            {
                file = "breast-cancer-assignment5.txt";
            }
            else if (x == "2")
            {
                file = "german-assignment5.txt";
            }
            else
            {
                file = x;
            }
            DataSet data = new DataSet();
            data.Input(file);
            PrintLog("数据集读取完毕...");
            ClassifierValidator validator = new ClassifierValidator(data, PrintLog);
            x = "";
            while (x!="1" && x!="2" && x!="3")
            {
                Console.WriteLine("输入需要使用的分类模式\n\t1:普通c4.5决策树\n\t2:随机森林\n\t3:带Adaboost的决策树");
                x = Console.ReadLine();
            }

            var result = validator.CrossValidation(10, Convert.ToInt32(x));
            Console.WriteLine("平均准确度" + result.Mean_accuracy.ToString("0.00%"));
            Console.WriteLine("准确度标准差" + result.SD_accuracy.ToString("0.00"));
        }

        static void Main(string[] args)
        {
            if (File.Exists("log.txt")) File.Delete("log.txt");//log文件
            CrossTest();
            Console.WriteLine("结束");
            Console.ReadLine();
        }
    }
}
