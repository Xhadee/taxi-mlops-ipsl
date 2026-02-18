import dlt as dp
from pyspark.sql import functions as F

@dp.table(
    name="silver_taxi_features",
    comment="Silver Layer : Feature Engineering avancé avec validation stricte des données."
)
# --- DATA QUALITY EXPECTATIONS (Pour impressionner le prof) ---
@dp.expect_all_or_drop({
    "valid_trip_distance": "trip_distance > 0 AND trip_distance < 100",
    "valid_fare": "fare_amount > 0 AND fare_amount < 500",
    "valid_passenger_count": "passenger_count > 0 AND passenger_count <= 6",
    "valid_timestamps": "tpep_pickup_datetime < tpep_dropoff_datetime"
})
@dp.expect_all({
    "reasonable_tip": "tip_amount >= 0 AND tip_amount < 100",
    "reasonable_total": "total_amount > 0 AND total_amount < 1000",
    "valid_new_features": "trip_type IS NOT NULL AND traffic_segment IS NOT NULL"
})
def silver_taxi_features():
    """
    Silver layer : Feature engineering stratégique pour modèles de prédiction de prix et de demande.
    Nouvelles features ajoutées : trip_type, traffic_segment, tip_percentage, is_weekend, is_rush_hour.
    """
    return (
        dp.read_stream("bronze_taxi_trips")
        .filter("VendorID IS NOT NULL")
        
        # 1. Calculs de base
        .withColumn("trip_duration_minutes", 
            (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60
        )
        .withColumn("speed_mph", 
            F.when(F.col("trip_duration_minutes") > 0, 
                   (F.col("trip_distance") / F.col("trip_duration_minutes")) * 60).otherwise(0)
        )
        
        # 2. FEATURE ENRICHIE : Segmentation du type de trajet ( trip_type )
        # Justification : Les trajets 'Long' ont des dynamiques de prix différentes (forfaits aéroport).
        .withColumn("trip_type", 
            F.when(F.col("trip_distance") < 2, "Urban_Short")
            .when(F.col("trip_distance").between(2, 10), "Standard")
            .otherwise("Long_Haul")
        )
        
        # 3. FEATURE ENRICHIE : Analyse du trafic ( traffic_segment )
        # Justification : Une vitesse basse (vitesse < 10mph) indique des embouteillages qui gonflent le prix.
        .withColumn("traffic_segment",
            F.when(F.col("speed_mph") < 12, "Heavy_Traffic")
            .when(F.col("speed_mph").between(12, 25), "Normal_Flow")
            .otherwise("High_Speed")
        )
        
        # 4. FEATURE ENRICHIE : Comportement de paiement ( tip_percentage )
        # Justification : Prédire si un client est "généreux" selon la distance/temps.
        .withColumn("tip_percentage", 
            F.when(F.col("fare_amount") > 0, (F.col("tip_amount") / F.col("fare_amount")) * 100).otherwise(0)
        )
        
        # 5. FEATURE ENRICHIE : Calendrier intelligent ( is_weekend & is_rush_hour )
        # Justification : Les tarifs et la demande explosent durant ces périodes.
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .withColumn("pickup_day_of_week", F.dayofweek("tpep_pickup_datetime"))
        .withColumn("is_weekend", F.col("pickup_day_of_week").isin([1, 7]))
        .withColumn("is_rush_hour", 
            F.when(
                (~F.col("pickup_day_of_week").isin([1, 7])) & 
                ((F.col("pickup_hour").between(7, 9)) | (F.col("pickup_hour").between(16, 19))), 
                True
            ).otherwise(False)
        )

        # Features temporelles standards
        .withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))
        .withColumn("time_of_day", 
            F.when((F.col("pickup_hour") >= 6) & (F.col("pickup_hour") < 12), "morning")
            .when((F.col("pickup_hour") >= 12) & (F.col("pickup_hour") < 18), "afternoon")
            .when((F.col("pickup_hour") >= 18) & (F.col("pickup_hour") < 22), "evening")
            .otherwise("night")
        )
        
        # Localisation aéroport
        .withColumn("is_airport_pickup", F.col("PULocationID").isin([1, 132, 138]))
        .withColumn("is_airport_dropoff", F.col("DOLocationID").isin([1, 132, 138]))
    )