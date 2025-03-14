# Otimização de Pipeline ETL e Machine Learning com PySpark
# Módulo de Processamento do Pipeline (ETL)

import os
import os.path
import numpy
from pyspark.ml.feature import * 
from pyspark.sql import functions
from pyspark.sql.functions import * 
from pyspark.sql.types import StringType,IntegerType
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from as_log import sil_grava_log

# Define uma função para calcular a quantidade e a porcentagem de valores nulos em cada coluna de um DataFrame
def sil_calcula_valores_nulos(df):
    
    null_columns_counts = []
    numRows = df.count()
    for k in df.columns:
        nullRows = df.where(col(k).isNull()).count()
        if(nullRows > 0):
            temp = k, nullRows, (nullRows / numRows) * 100
            null_columns_counts.append(temp)
    return null_columns_counts

# Função para limpeza e transformação
def sil_limpa_transforma_dados(spark):
	path = "/opt/spark/data/"
	sil_grava_log("Importando os dados...")
	reviews = spark.read.csv(path + 'dataset.csv', header=True, escape="\"")
	sil_grava_log("Dados Importados com Sucesso.")
	sil_grava_log("Total de Registros: " + str(reviews.count()))
	sil_grava_log("Verificando se Existem Dados Nulos.")

	# Valores ausentes
	null_columns_calc_list = sil_calcula_valores_nulos(reviews)
	if (len(null_columns_calc_list) > 0):
		for column in null_columns_calc_list:
			sil_grava_log("Coluna " + str(column[0]) + " possui " + str(column[2]) + " de dados nulos")
		reviews = reviews.dropna()
		sil_grava_log("Dados nulos excluídos")
		sil_grava_log("Total de Registros Depois da Limpeza: " + str(reviews.count()))
	else:
		sil_grava_log("Valores Ausentes Nao Foram Detectados.")

	sil_grava_log("Verificando o Balanceamento de Classes.")
	
	# Conta os registros de avaliações positivas e negativas
	count_positive_sentiment = reviews.where(reviews['sentiment'] == "positive").count()
	count_negative_sentiment = reviews.where(reviews['sentiment'] == "negative").count()

	sil_grava_log("Existem " + str(count_positive_sentiment) + " reviews positivos e " + str(count_negative_sentiment) + " reviews negativos.")

	df = reviews

	sil_grava_log("Transformando os Dados.")
	
	# Cria e treina o indexador
	indexer = StringIndexer(inputCol="sentiment", outputCol="label")
	df = indexer.fit(df).transform(df)

	sil_grava_log(" Limpeza dos Dados.")
	
	# Remove caracteres especiais dos dados de texto
	df = df.withColumn("review", regexp_replace(df["review"], '<.*/>', ''))
	df = df.withColumn("review", regexp_replace(df["review"], '[^A-Za-z ]+', ''))
	df = df.withColumn("review", regexp_replace(df["review"], ' +', ' '))
	df = df.withColumn("review", lower(df["review"]))

	sil_grava_log(" Os Dados de Texto Foram Limpos.")
	sil_grava_log(" Tokenizando os Dados de Texto.")

	# Cria e aplica o tokenizador 
	regex_tokenizer = RegexTokenizer(inputCol="review", outputCol="words", pattern="\\W")
	df = regex_tokenizer.transform(df)
	sil_grava_log(" Removendo Stop Words.")

	# Criae aplica o objeto para remover stop words
	remover = StopWordsRemover(inputCol="words", outputCol="filtered")
	feature_data = remover.transform(df)
	sil_grava_log(" Aplicando HashingTF.")

	# Cria e aplica o processador de texto 1
	hashingTF = HashingTF(inputCol="filtered", outputCol="rawfeatures", numFeatures=250)
	HTFfeaturizedData = hashingTF.transform(feature_data)
	sil_grava_log(" Aplicando IDF.")

	# Cria e aplica o processador de texto 2
	idf = IDF(inputCol="rawfeatures", outputCol="features")
	idfModel = idf.fit(HTFfeaturizedData)
	TFIDFfeaturizedData = idfModel.transform(HTFfeaturizedData)
	
	# Ajusta o nome dos objetos
	TFIDFfeaturizedData.name = 'TFIDFfeaturizedData'
	HTFfeaturizedData = HTFfeaturizedData.withColumnRenamed("rawfeatures","features")
	HTFfeaturizedData.name = 'HTFfeaturizedData' 
	sil_grava_log(" Aplicando Word2Vec.")

	# Cria e aplica o processador de texto 3
	word2Vec = Word2Vec(vectorSize=250, minCount=5, inputCol="filtered", outputCol="features")
	model = word2Vec.fit(feature_data)
	W2VfeaturizedData = model.transform(feature_data)
	sil_grava_log(" Padronizando os Dados com MinMaxScaler.")

	# Cria e aplica o padronizador
	scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
	scalerModel = scaler.fit(W2VfeaturizedData)
	scaled_data = scalerModel.transform(W2VfeaturizedData)
	
	# Ajusta o nome dos objetos
	W2VfeaturizedData = scaled_data.select('sentiment','review','label','scaledFeatures')
	W2VfeaturizedData = W2VfeaturizedData.withColumnRenamed('scaledFeatures','features')
	W2VfeaturizedData.name = 'W2VfeaturizedData'

	sil_grava_log(" Salvando os Dados Limpos e Transformados.")

	path = '/opt/spark/data/dados_processados/'
	HTFfeaturizedData.write.mode("Overwrite").partitionBy("label").parquet(path)
	TFIDFfeaturizedData.write.mode("Overwrite").partitionBy("label").parquet(path)
	W2VfeaturizedData.write.mode("Overwrite").partitionBy("label").parquet(path)

	sil_grava_log(" Dados Salvos com Sucesso.")

	return HTFfeaturizedData, TFIDFfeaturizedData, W2VfeaturizedData

	