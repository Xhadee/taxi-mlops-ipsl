import dlt as dp

@dp.table(
    comment="Raw taxi trip data ingested from cloud storage using Auto Loader"
)
def bronze_taxi_trips():
    """
    Bronze layer: Raw taxi trip data ingestion
    Utilise le chemin officiel partagé du catalogue de production.
    """
    # mon chemin que j'ai copy depuis le catalogue
    source_path = "/Volumes/taxi_mlops_prod/taxi_analytics/yellowdata"
    
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(source_path)
    )