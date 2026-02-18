import dlt
from pyspark.sql import functions as F

@dlt.table(
    name="ml_model_registry",
    comment="Registre des métadonnées du modèle incluant la comparaison des performances"
)
def ml_model_registry():
    """
    ML Model Registry - Version Améliorée
    Enregistre les performances du modèle de base vs le modèle enrichi.
    """
    # Lecture des métriques de comparaison
    metrics_df = dlt.read("ml_model_training")
    
    return (
        metrics_df
        .withColumn("model_name", F.lit("taxi_fare_prediction_model"))
        .withColumn("model_version", F.lit("v2.0_improved")) # On passe en v2 !
        .withColumn("model_status", F.lit("active"))
        # Ajout manuel de model_type pour corriger l'erreur
        .withColumn("model_type", F.lit("Heuristic Linear Regression"))
        .withColumn("catalog", F.lit("taxi_mlops_prod"))
        .withColumn("schema", F.lit("khady_ndiaye"))
        .withColumn("description", F.lit("Modèle amélioré avec features de trafic et rush hour (Partie 3)"))
        .withColumn("registered_at", F.current_timestamp())
        .select(
            "model_name",
            "model_version",
            "model_type",       
            "model_status",
            "catalog",
            "schema",
            "rmse_base",        # Colonnes provenant de ml_model_training amélioré
            "rmse_improved",
            "mae_base",
            "mae_improved",
            "performance_gain_pct",
            "description",
            "registered_at"
        )
    )