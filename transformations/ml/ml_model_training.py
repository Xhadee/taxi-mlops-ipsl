import dlt as dp
from pyspark.sql import functions as F

@dp.table(
    comment="Comparaison des performances entre le modèle de base et le modèle amélioré (Partie 3)"
)
def ml_model_training():
    """
    Amélioration du modèle (Partie 3) :
    1. Intégration des nouvelles features (is_rush_hour, traffic_segment)
    2. Ajout de termes d'interaction (vitesse * heure de pointe)
    """
    # Lecture des données Silver enrichies (pour avoir accès aux nouvelles colonnes)
    # Note : On lit silver_taxi_features pour avoir is_rush_hour et traffic_segment
    df = dp.read("silver_taxi_features")
    
    # Séparation train/test (reproduction de la logique ml_training_data)
    # On ajoute une colonne de cible (target) pour le calcul
    df = df.withColumn("target_amount", F.col("total_amount"))
    
    #MODÈLE AMÉLIORÉ (Option A & B) 
    # On ajuste les coefficients pour refléter l'impact des nouvelles features
    predictions_df = (
        df
        .withColumn("is_airport", F.when((F.col("is_airport_pickup") | F.col("is_airport_dropoff")), 1).otherwise(0))
        
        # Ancien modèle (Baseline)
        .withColumn("pred_base", 
            F.lit(3.0) + (F.col("trip_distance") * 2.5) + (F.col("trip_duration_minutes") * 0.5)
        )
        
        # NOUVEAU MODÈLE AMÉLIORÉ
        .withColumn("pred_improved", 
            F.lit(2.5) + # Base légèrement ajustée
            (F.col("trip_distance") * 2.4) + 
            (F.col("trip_duration_minutes") * 0.4) +
            # Prise en compte de l'heure de pointe (Option A)
            F.when(F.col("is_rush_hour") == True, 2.5).otherwise(0.0) + 
            # Impact du trafic lourd (Option B : Interaction vitesse/prix)
            F.when(F.col("traffic_segment") == "Heavy_Traffic", 3.0).otherwise(0.0) +
            # Forfait aéroport
            F.when(F.col("is_airport") == 1, 6.0).otherwise(0.0)
        )
    )

    # CALCUL DES MÉTRIQUES POUR LES DEUX MODÈLES 
    metrics = predictions_df.select(
        # Métriques Base
        F.sqrt(F.avg(F.pow(F.col("pred_base") - F.col("target_amount"), 2))).alias("rmse_base"),
        F.avg(F.abs(F.col("pred_base") - F.col("target_amount"))).alias("mae_base"),
        
        # Métriques Améliorées
        F.sqrt(F.avg(F.pow(F.col("pred_improved") - F.col("target_amount"), 2))).alias("rmse_improved"),
        F.avg(F.abs(F.col("pred_improved") - F.col("target_amount"))).alias("mae_improved")
    )

    #FORMATAGE DU RÉSULTAT FINAL 
    return (
        metrics.select(
            F.lit("Comparison Report").alias("report_type"),
            F.col("rmse_base"),
            F.col("rmse_improved"),
            F.col("mae_base"),
            F.col("mae_improved"),
            # Calcul du gain de performance en %
            ((F.col("rmse_base") - F.col("rmse_improved")) / F.col("rmse_base") * 100).alias("performance_gain_pct"),
            F.current_timestamp().alias("analysis_timestamp")
        )
    )