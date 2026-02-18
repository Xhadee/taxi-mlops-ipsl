import dlt as dp
from pyspark.sql import functions as F

@dp.table(
    comment="Model training results and evaluation metrics for taxi fare prediction"
)
def ml_model_training():
    """
    ML Model Training and Validation
    Calcule les métriques de performance (RMSE, MAE) du modèle.
    """
    # Lecture des données depuis la table GOLD (ml_training_data)
    df = dp.read("ml_training_data")
    
    # Séparation train/test
    train_df = df.filter("is_training = true")
    test_df = df.filter("is_training = false")
    
    # Encodage des variables (One-Hot Encoding simplifié)
    test_encoded = (
        test_df
        .withColumn("time_evening", F.when(F.col("time_of_day") == "evening", 1.0).otherwise(0.0))
        .withColumn("time_night", F.when(F.col("time_of_day") == "night", 1.0).otherwise(0.0))
        .withColumn("airport_pickup_flag", F.col("is_airport_pickup").cast("double"))
        .withColumn("airport_dropoff_flag", F.col("is_airport_dropoff").cast("double"))
    )
    
    # Application du modèle pour obtenir les prédictions
    predictions = (
        test_encoded
        .withColumn(
            "predicted_total_amount",
            F.lit(3.0) +
            (F.col("trip_distance") * 2.5) +
            (F.col("trip_duration_minutes") * 0.5) +
            F.when(F.col("time_evening") == 1, 2.0).otherwise(0.0) +
            F.when(F.col("time_night") == 1, 3.0).otherwise(0.0) +
            F.when(F.col("airport_pickup_flag") == 1, 5.0).otherwise(0.0) +
            F.when(F.col("airport_dropoff_flag") == 1, 5.0).otherwise(0.0) +
            (F.col("passenger_count") * 0.5)
        )
    )
    
    # Calcul des métriques d'évaluation
    metrics = predictions.agg(
        F.sqrt(F.avg(F.pow(F.col("predicted_total_amount") - F.col("target_total_amount"), 2))).alias("rmse"),
        F.avg(F.abs(F.col("predicted_total_amount") - F.col("target_total_amount"))).alias("mae"),
        F.corr("predicted_total_amount", "target_total_amount").alias("correlation")
    )
    
    # Retourne les résultats finaux
    return metrics.select(
        F.lit("simple_linear_model").alias("model_type"),
        F.col("rmse"),
        F.col("mae"),
        F.col("correlation"),
        F.current_timestamp().alias("training_timestamp")
    )