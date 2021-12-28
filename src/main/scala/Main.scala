import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.ml.linalg.{SparseVector => SV}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.parsing.json.JSON

object Main {
  def main(args: Array[String]) {
    /*初始化SparkContext*/
    val conf = new SparkConf().setMaster("local").setAppName("paperAnalysis") //创建SparkConf
    //val conf = new SparkConf().setMaster("master").setAppName("paperAnalysis") //创建SparkConf
    val spark = new SparkContext(conf) //基于SparkConf创建一个SparkContext对象
    val sqlContext = new SQLContext(spark)

    val raw_data = spark.textFile("D:/sparkProject/paperAnalysis/input/paper.txt")  //读取本地文件
    //val raw_data = spark.textFile("hdfs://master:9000/input/paper.txt")  //读取hdfs文件

    //将原始数据解析为json格式，从中提取abstract
    val paper_raw = raw_data.map(JSON.parseFull).map(r => {
      val abs = r.get.asInstanceOf[Map[String, Any]].get("abstract").get.asInstanceOf[String]
      val title = r.get.asInstanceOf[Map[String, Any]].get("title").get.asInstanceOf[String]
      (title, abs)
    })

    var title = ""
    args.foreach(x => title = title.concat(" ").concat(x))
    title = title.replaceFirst(" ", "")
    //判断是否存在指定文章
    if(paper_raw.filter(x => x._1 == title).count() != 1){
      print("No paper titled \"" + title + "\" found!\r\n")
      spark.stop()
      return
    }

    //停用词集合
    val stop_words = Set(
      "the","a","an","of","or","in","for","by","on","but", "is", "not", "with", "as", "was", "if",
      "they", "are", "this", "and", "it", "have", "from", "at", "my", "be", "that", "to"
    )
    //对摘要以及其中的单词进行过滤
    val paper_filtered = paper_raw.map(x => {
      var abs = x._2.replaceAll("\n", " ")    //替代字符串中\n
        .replaceAll(" +", " ")    //替代字符串中连续空格
      if(abs.charAt(0) == ' '){
        abs = abs.replaceFirst(" ", "")    //替代字符串中首字符空格
      }
      val words = x._2.split("""\W+""").map(_.toLowerCase())    //切割摘要为词汇
        .filter(token => """[^0-9]*""".r.pattern.matcher(token).matches())   //筛掉数字和包含数字的词
        .filterNot(token => stop_words.contains(token))    //筛掉停用词
        .filter(token => token.size >= 2)   //筛掉长度过短的词
      (x._1, abs, words)
    })

    //rdd转为dataframe
    import sqlContext.implicits._
    val paper_df = paper_filtered.toDF("title", "abstract", "words")

    //hashing计算tf值，同时过滤掉停用词
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
    val featured_data = hashingTF.transform(paper_df)
    featured_data.cache()

    //为每个单词计算逆向频率，并将词频向量转化为TF-IDF向量
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featured_data)
    val rescaled_data = idfModel.transform(featured_data)
//    rescaled_data.show()

    //将每篇文章的TF-IDF向量提取出来
    val paper_vector = rescaled_data.select("title","abstract", "features").rdd.map(x => {
      val r = Row.unapplySeq(x).get.toList
      val title = r(0).asInstanceOf[String]
      val abs = r(1).asInstanceOf[String]
      val vector = r(2).asInstanceOf[SV]
      (title, abs, vector)
    })

    //计算每篇文章和指定文章的向量余弦，选择最相似的10篇文章
    val vec = paper_vector.filter(x => x._1 == title).take(1)(0)._3
    val v1 = new SparseVector(vec.size, vec.indices, vec.values)
    val result_array = paper_vector.map(x => {
      val v2 = new SparseVector(x._3.size, x._3.indices, x._3.values)
      val t1 = math.sqrt(v1.values.map(num => math.pow(num, 2)).sum)
      val t2 = math.sqrt(v2.values.map(num => math.pow(num, 2)).sum)
      (v1.dot(v2)/t1/t2, x._1, x._2)
    }).sortBy(_._1, false).take(10)

    //将及结果以df形式展示
    val result_df = spark.parallelize(result_array).toDF("similarity", "title", "abstract")
    result_df.show(false)
  }
}
