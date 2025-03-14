# Otimização de Pipeline ETL e Machine Learning com PySpark
# Módulo Principal do Pipeline 

import os
import traceback
import pyspark 
from pyspark.sql import SparkSession
from as_log import sil_grava_log
from as_processamento import sil_limpa_transforma_dados
from as_ml import sil_cria_modelos_ml

# Cria a sessão e grava o log no caso de erro
try:
	spark = SparkSession.builder.appName("analise sentimento").getOrCreate()
	spark.sparkContext.setLogLevel("ERROR")
except:
	sil_grava_log("Ocorreu uma falha na Inicialização do Spark.")
	sil_grava_log(traceback.format_exc())
	raise Exception(traceback.format_exc())

sil_grava_log("\nIniciando a analise de sentimento.")
sil_grava_log("Spark Inicializado.")

# Bloco de limpeza e transformação
try:
	DadosHTFfeaturized, DadosTFIDFfeaturized, DadosW2Vfeaturized = sil_limpa_transforma_dados(spark)
except:
	sil_grava_log("Log Ocorreu uma falha na limpeza e transformação dos dados.")
	sil_grava_log(traceback.format_exc())
	spark.stop()
	raise Exception(traceback.format_exc())

# Bloco de criação dos modelos de Machine Learning
try:
	sil_cria_modelos_ml(spark, DadosHTFfeaturized, DadosTFIDFfeaturized, DadosW2Vfeaturized)
except:
	sil_grava_log("Log Ocorreu Alguma Falha ao Criar os Modelos de Machine Learning.")
	sil_grava_log(traceback.format_exc())
	spark.stop()
	raise Exception(traceback.format_exc())

sil_grava_log("Log Modelos Criados e Salvos com Sucesso.")
sil_grava_log("Log Processamento Finalizado com Sucesso.\n")
spark.stop()



