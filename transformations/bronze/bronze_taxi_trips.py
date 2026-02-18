import dlt as dp

@dp.table(
    name="bronze_taxi_trips",
    comment="Raw taxi trip data ingested from cloud storage using Auto Loader"
)
def bronze_taxi_trips():
    """
    Bronze layer: Raw taxi trip data ingestion
    Correction : Ajout de la gestion du format timestampNtz pour Delta Lake.
    """
    source_path = "/Volumes/taxi_mlops_prod/taxi_analytics/yellowdata"
    
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        # Ces deux options règlent le problème du timestampNtz
        .option("spark.sql.parquet.v2.datetimeRebaseModeInRead", "CORRECTED")
        .option("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(source_path)
    )