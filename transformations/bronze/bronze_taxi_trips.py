import dlt as dp
from pyspark.sql import functions as F

@dp.table(
    name="bronze_taxi_trips",
    comment="Raw taxi trip data ingested from cloud storage using Auto Loader"
)
def bronze_taxi_trips():
    """
    Bronze layer : Ingestion avec correction du type timestampNtz.
    """
    source_path = "/Volumes/taxi_mlops_prod/taxi_analytics/yellowdata"
    
    # On définit explicitement les options pour éviter le conflit de schéma
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        # On force Spark à utiliser le format de timestamp standard (LTZ)
        .option("spark.sql.parquet.v2.datetimeRebaseModeInRead", "CORRECTED")
        .option("cloudFiles.inferColumnTypes", "true")
        # Cette option force le schéma à rester compatible avec les tables Delta standards
        .option("cloudFiles.schemaEvolutionMode", "rescue")
        .load(source_path)
        # Sécurité supplémentaire : on convertit explicitement les colonnes de temps
        .withColumn("tpep_pickup_datetime", F.col("tpep_pickup_datetime").cast("timestamp"))
        .withColumn("tpep_dropoff_datetime", F.col("tpep_dropoff_datetime").cast("timestamp"))
    )