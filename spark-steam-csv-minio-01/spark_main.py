import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import os

cwd = os.getcwd()

# .config("spark.jars", "spark_jars/hadoop-aws-2.7.3.jar") \
#.config('spark.jars.packages','org.apache.hadoop:hadoop-aws:2.7.3')\
#.config("spark.executor.extraClassPath", f"{cwd}/spark_jars/hadoop-aws-2.7.3.jar")\
#.config("spark.jars", f"{cwd}/spark_jars/hadoop-aws-2.7.3.jar")\
#.config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.3")\
#.config('spark.jars.packages','org.apache.hadoop:hadoop-aws:3.3.6')\

# os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.hadoop:hadoop-aws:3.3.6 pyspark-shell'
#.config('spark.hadoop.fs.s3a.aws.credentials.provider', 'org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider')\
#.config('spark.hadoop.fs.s3a.aws.credentials.provider', 'org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider')\


minio_url = "minio-service.kubeflow.svc.cluster.local:9000"
spark = SparkSession\
.builder\
.appName("ReadTextFilesFromS3")\
.master("local[*]")\
.config('spark.jars.packages','org.apache.hadoop:hadoop-aws:3.3.4')\
.config("spark.hadoop.fs.s3a.endpoint", "http://"+minio_url)\
.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")\
.config("spark.hadoop.fs.s3a.path.style.access", "true")\
.config("spark.hadoop.fs.s3a.access.key", "minio")\
.config("spark.hadoop.fs.s3a.secret.key", "minio123")\
.getOrCreate()



sc = spark.sparkContext
jars = sc._jsc.sc().listJars()
print("recognized additional jars:",jars)

local_file_path="random_data.csv"
minio_file_path = "s3a://jupyter-spark-01/random_data.csv"
print(minio_file_path)
# df = spark.read.csv(minio_file_path)
df = spark.read.csv(minio_file_path,header=False)
df.show()
