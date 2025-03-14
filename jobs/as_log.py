# Otimização de Pipeline ETL e Machine Learning com PySpark
# Módulo de Log

import os
import os.path
import pendulum
import traceback

# Define a função sil_grava_log que recebe um texto como parâmetro, que será gravado no log
def sil_grava_log(texto):
    path = "/opt/spark/data"
    agora = pendulum.now()
    data_arquivo = agora.format('YYYYMMDD')
    data_hora_log = agora.format('YYYY-MM-DD HH:mm:ss')
    nome_arquivo = path + data_arquivo + "-log_spark_analise_sentimento.txt"
    texto_log = ''

    try:
        if os.path.isfile(nome_arquivo):  
            arquivo = open(nome_arquivo, "a")
            texto_log = texto_log + '\n'  

        else:
            arquivo = open(nome_arquivo, "w")  

    except:
        print("Erro na tentativa de acessar o arquivo para criar os logs.")
        raise Exception(traceback.format_exc())  

    texto_log = texto_log + "[" + data_hora_log + "] - " + texto

    arquivo.write(texto_log)  

    # obs: Tirar para automatizar
    print(texto)  
    
    arquivo.close()  
    
