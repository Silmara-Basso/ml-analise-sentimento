# Otimização de Pipeline ETL e Machine Learning com PySpark
# Módulo de Machine Learning

import os
import numpy
from pyspark.ml.feature import * 
from pyspark.sql import functions
from pyspark.sql.functions import * 
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from as_log import sil_grava_log

# Classe para treinar e avaliar o modelo
def silTreinaAvaliaModelo(spark, classifier, features, classes, train, test):

    # Método para definir o tipo de classificador
    def FindMtype(classifier):
        M = classifier
        Mtype = type(M).__name__
        return Mtype
    Mtype = FindMtype(classifier)
    
    # Método para o treinamento do modelo
    def IntanceFitModel(Mtype, classifier, classes, features, train):
        
        if Mtype in("LogisticRegression"):
  
            # Grid de hiperparâmetros para otimização
            paramGrid = (ParamGridBuilder().addGrid(classifier.maxIter, [10, 15, 20]).build())
            
            # Validação cruzada para otimização de hiperparâmetros
            crossval = CrossValidator(estimator = classifier,
                                      estimatorParamMaps = paramGrid,
                                      evaluator = MulticlassClassificationEvaluator(),
                                      numFolds = 2)

            # Cria objeto de treinamento
            fitModel = crossval.fit(train)

            return fitModel
    
    fitModel = IntanceFitModel(Mtype, classifier, classes, features, train)
    
    # Imprime algumas métricas
    if fitModel is not None:

        if Mtype in("LogisticRegression"):
            BestModel = fitModel.bestModel
            sil_grava_log(Mtype)
            global LR_coefficients
            LR_coefficients = BestModel.coefficientMatrix.toArray()
            global LR_BestModel
            LR_BestModel = BestModel
        
    columns = ['Classifier', 'Result']
    
    # Extrai previsões do modelo com dados de teste
    predictions = fitModel.transform(test)
    
    # Cria o avaliador
    MC_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    
    # Calcula a acurácia
    accuracy = (MC_evaluator.evaluate(predictions)) * 100
    
    sil_grava_log( "Classificador: " + Mtype + " / Acuracia: " + str(accuracy))

    Mtype = [Mtype]
    score = [str(accuracy)]
    result = spark.createDataFrame(zip(Mtype,score), schema=columns)
    result = result.withColumn('Result',result.Result.substr(0, 5))
    path = "/opt/spark/data/modelos/"

    fitModel.write().overwrite().save(path)

    sil_grava_log("Modelo Salvo com Sucesso.")
    
    return result

# Função para criar o modelo de Machine Learning
def sil_cria_modelos_ml(spark, HTFfeaturizedData, TFIDFfeaturizedData, W2VfeaturizedData):

    # apenas um classificador, mas é possível incluir outros
    classifiers = [LogisticRegression()] 

    featureDF_list = [HTFfeaturizedData, TFIDFfeaturizedData, W2VfeaturizedData]

    for featureDF in featureDF_list:
        sil_grava_log(featureDF.name + " Resultados: ")
        train, test = featureDF.randomSplit([0.7, 0.3],seed = 11)
        train.name = featureDF.name
        features = featureDF.select(['features']).collect()
        classes = featureDF.select("label").distinct().count()
        columns = ['Classifier', 'Result']
        vals = [("Place Holder","N/A")]
        results = spark.createDataFrame(vals, columns)

        for classifier in classifiers:
            new_result = silTreinaAvaliaModelo(spark,
                                               classifier,
                                               features,
                                               classes,
                                               train,
                                               test)

            results = results.union(new_result)
            results = results.where("Classifier!='Place Holder'")
