import dlt
from pyspark.sql import functions as F

@dlt.table(
    name="ml_model_registry",
    comment="Model metadata and registration information"
)
def ml_model_registry():
    """
    ML Model Registry
    Stocke les métadonnées et le statut du modèle.
    Lit directement depuis la fonction 'ml_model_training'.
    """
    # On lit les métriques générées à l'étape précédente
    metrics_df = dlt.read("ml_model_training")
    
    # Création de l'entrée du registre avec les métadonnées
    return (
        metrics_df
        .withColumn("model_name", F.lit("taxi_fare_prediction_model"))
        .withColumn("model_version", F.lit("v1.0"))
        .withColumn("model_status", F.lit("active"))
        # On met tes informations de catalogue et schéma ici
        .withColumn("catalog", F.lit("workspace"))
        .withColumn("schema", F.lit("khady_ndiaye"))
        .withColumn("description", F.lit("Linear regression model for taxi fare prediction based on trip characteristics"))
        .withColumn("registered_at", F.current_timestamp())
        .select(
            "model_name",
            "model_version",
            "model_type",
            "model_status",
            "catalog",
            "schema",
            "rmse",
            "mae",
            "correlation",
            "description",
            "training_timestamp",
            "registered_at"
        )
    )