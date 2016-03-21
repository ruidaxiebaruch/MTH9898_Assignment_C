import os
import random
import sys
import csv as csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random as rd
import pandas.io.data as web
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from datetime import datetime

APP_NAME = "My Spark Application"

def getCompanyID(strs):
    companyDict = {'TripAdvisor':"0", 'AFLAC':"1"}

    res = [];
    sCol = strs.split(" ")
    for s in sCol:
        if companyDict.has_key(s.lower()):
            res.append(companyDict[s.lower()])
    return res;

def stringDate(x):
    month = {"Jan":"01", "Feb":"02", "Mar":"03", "Apr":"04", "May":"05", "Jun":"06", "Jul":"07", "Aug":"08",
            "Sep":"09", "Oct":"10", "Nov":"11", "Dec":"12"}
    xSplit = x.split(" ")
    yyyy = xSplit[5]
    mm = month[xSplit[1]]
    dd = xSplit[2]
    # to put the dates in the format yyyy-mm-dd
    return "-".join([yyyy, mm, dd])

def maps(x):
    timeStamp = stringDate(x.created_at)
    companyID = getCompanyID(x.text)
    res = []
    for id in companyID:
        res.append(timeStamp + "\t" + id + "\t" + x.text)
    return res

if __name__ == "__main__":
    Conf = SparkConf().setAppName(APP_NAME)
    Conf = Conf.setMaster("local[*]")
    sc = SparkContext(Conf = Conf)
    sqlContext = SQLContext(sc)
    dataFile = sqlContext.read.json("tweets/*.tar.gz")
    dataFile = dataFile.filter(dataFile["lang"].like('%en%'))
    dataFile = dataFile.select(["created_at", "text"])
    rdd = dataFile.rdd
    rdd1 = rdd.flatMap(maps)
    rdd1.saveAsTextFile("tweet4")

syms = ['TRIP', 'AFL']
start = datetime(2012,12,31)
end = datetime(2014,1,1)
stockRaw = web.DataReader(syms, 'yahoo', start, end)
sliceKey = 'Adj Close'
adjClose = stockRaw.ix[sliceKey]
adjClose = adjClose.shift(1) / adjClose - 1
adjClose.to_csv("stockreturn.txt")

APP_NAME = "My Spark Application"
def mapp(x):
    compdict = {'0':"tripadvisor", '1':"aflac"}
    dict2={'boeing':1,'google':3,'amd':0,'dupont':2,'intel':4,'pfizer':7,'merck':5,'nike':6,'verzion':8}
    context=x[1][0]
    return3=x[1][1]
    context2=context.split("\t")[1]
    key=context.split("\t")[0]
    return2=return3.split(",")[dict2[compdict[key]]]
    res="\t".join([x[0],key,return2,context2])
    return res
if __name__=="__main__":
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("local[*]")
    sc = SparkContext(conf=conf)
    df = sc.textFile("tweets/*.txt")
    parts = df.map(lambda x: (x.split(",")[0], ",".join(x.split(",")[1:])))
    df2=sc.textFile("tweet4")
    part2=df2.map(lambda l: ((l.split("\t")[0],"\t".join(l.split("\t")[1:]))))
    part3=part2.join(parts)
    part4=part3.map(mapp)
    part4.saveAsTextFile("tweet5")

# number of stocks under consideration
nStocks = 2

# dictionary of sentimental words
sentimentDict = {}

# extract word
def obtainFeat(content):
    def work(content, dict):
        colByTab = content.split("\t")
        col = colByTab[3].split(" ")
        retValue = colByTab[2]
        stockID = int(colByTab[1])
        res = [0.0, 0.0] * nStocks
        for w in col:
            if sentimentDict.has_key(w.lower()):
                if sentimentDict[w.lower()] == 1:
                    res[2 * stockID] = res[2 * stockID] + 1
            if sentimentDict[w.lower()] == -1:
                res[1 + 2 * stockID] = res[1 + 2 * stockID] + 1
            res[2 * stockID] = res[2*stockID] / (1.0+len(col))
            res[1 + 2 * stockID] = res[1 + 2 * stockID] / (1.0 + len(col))
            if res[2 * stockID] == 0 and res[2 * stockID] == 0:
                res[2 * stockID] = random.random() / 10
            ret = " ".join([str(x) for x in res])
        return str(retValue) + " " + ret
    return work(content, sentimentDict)


if __name__=="__main__":
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("local[*]")
    sc = SparkContext(conf=conf)
    tweetData = sc.textFile("intermFile2");
    
    sentimentDictSource = sc.textFile("DictionaryX");
    for p in sentimentDictSource.collect():
        cols = p.split("\t")
        sentimentDict[cols[0]] = cols[1]
    
    res=tweetData.map(obtainFeat)
    res.saveAsTextFile("intermFile3")

APP_NAME2 = "sentiment_regression"
#stock number
stock_cnt = 9

# parse the data by blanks
def parseBlank(ss):
    line = ss.split(" ")
    values = [float(x) for x in line]
    return LabeledPoint(values[0], values[1:])

if __name__ == "__main__":
    conf = SparkConf().setAppName(APP_NAME2)
    conf = conf.setMaster("local[*]")
    sc = SparkContext(conf=conf)
    
    data = sc.textFile("intermFile3")
    training = data.map(parseBlank)
    
    # compute the simple regression
    model = LinearRegressionWithSGD.train(training,100,0.1)
    
    # Evaluate the model on training data
    valuesAndPreds = training.map(lambda p: (p.label, model.predict(p.features)))
    train_res = valuesAndPreds.map(lambda (v, p) : ((v - p) * (v - p))).reduce(lambda x, y: x + y) / valuesAndPreds.count()
    train_res.saveAsTextFile("intermFile4")


