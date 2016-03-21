from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
import sys
import os
import random
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
APP_NAME = "sentiment_regression"
#stock number
stock_cnt = 9

# parse the data
def parsePoint(ss):
    line = ss.split(" ")
    values = [float(x) for x in line]
    return LabeledPoint(values[0], values[1:])

if __name__ == "__main__":
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("local[*]")
    sc = SparkContext(conf=conf)

    data = sc.textFile("step3")
    train_data = data.map(parsePoint)

    # compute the simple regression
    model = LinearRegressionWithSGD.train(train_data,100,0.1)

    # Evaluate the model on training data
    valuesAndPreds = train_data.map(lambda p: (p.label, model.predict(p.features)))
    train_res = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
    train_res.saveAsTextFile("step4")
