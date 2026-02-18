import dlt as dp
from pyspark.sql import functions as F

@dp.table(
    comment="Batch predictions on 1000 sample taxi trips with actual vs predicted fare comparison"
)
def ml_predictions():
    """
    ML Batch Inference
    Applique la logique du modèle sur les données de test.
    Note: Lit directement depuis la fonction 'ml_training_data'.
    """
    # Récupère 1000 lignes de test depuis la table Gold
    test_data = (
        dp.read("ml_training_data")
        .filter("is_training = false")
        .limit(1000)
    )
    
    # Applique la logique de calcul du prix (modèle)
    predictions = (
        test_data
        # Encodage du moment de la journée
        .withColumn("time_morning", F.when(F.col("time_of_day") == "morning", 1.0).otherwise(0.0))
        .withColumn("time_afternoon", F.when(F.col("time_of_day") == "afternoon", 1.0).otherwise(0.0))
        .withColumn("time_evening", F.when(F.col("time_of_day") == "evening", 1.0).otherwise(0.0))
        .withColumn("time_night", F.when(F.col("time_of_day") == "night", 1.0).otherwise(0.0))
        
        # Encodage du type de paiement
        .withColumn("payment_credit", F.when(F.col("payment_type") == 1, 1.0).otherwise(0.0))
        .withColumn("payment_cash", F.when(F.col("payment_type") == 2, 1.0).otherwise(0.0))
        
        # Conversion booléen en numérique
        .withColumn("airport_pickup_flag", F.col("is_airport_pickup").cast("double"))
        .withColumn("airport_dropoff_flag", F.col("is_airport_dropoff").cast("double"))
        
        # Application de la formule du modèle (Coefficients)
        .withColumn(
            "predicted_total_amount",
            F.lit(3.0) + # Frais de base
            (F.col("trip_distance") * 2.5) +
            (F.col("trip_duration_minutes") * 0.5) +
            F.when(F.col("time_evening") == 1, 2.0).otherwise(0.0) +
            F.when(F.col("time_night") == 1, 3.0).otherwise(0.0) +
            F.when(F.col("airport_pickup_flag") == 1, 5.0).otherwise(0.0) +
            F.when(F.col("airport_dropoff_flag") == 1, 5.0).otherwise(0.0) +
            (F.col("passenger_count") * 0.5)
        )
        
        # Calcul des erreurs
        .withColumn("prediction_error", F.col("predicted_total_amount") - F.col("target_total_amount"))
        .withColumn("absolute_error", F.abs(F.col("prediction_error")))
        .withColumn("error_percentage", (F.col("absolute_error") / F.col("target_total_amount")) * 100)
    )
    
    return predictions.select(
        "trip_distance",
        "trip_duration_minutes",
        "speed_mph",
        "pickup_hour",
        "pickup_day_of_week",
        "time_of_day",
        "PULocationID",
        "DOLocationID",
        "is_airport_pickup",
        "is_airport_dropoff",
        "passenger_count",
        F.col("target_total_amount").alias("actual_total_amount"),
        "predicted_total_amount",
        "prediction_error",
        "absolute_error",
        "error_percentage"
    )