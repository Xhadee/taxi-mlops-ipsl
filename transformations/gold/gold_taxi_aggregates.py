import dlt
from pyspark.sql import functions as F

@dlt.table(
    name="gold_hourly_location_metrics",
    comment="Hourly aggregated taxi metrics by pickup location",
    partition_cols=["pickup_date"]
)
def gold_hourly_location_metrics():
    """
    Gold layer: Aggrégations horaires par localisation.
    Lit depuis la table silver_taxi_features.
    """
    return (
        dlt.read("silver_taxi_features")
        .groupBy(
            "pickup_date",
            "pickup_hour",
            "PULocationID"
        )
        .agg(
            F.count("*").alias("trip_count"),
            F.avg("trip_distance").alias("avg_trip_distance"),
            F.avg("trip_duration_minutes").alias("avg_trip_duration"),
            F.avg("fare_amount").alias("avg_fare_amount"),
            F.avg("total_amount").alias("avg_total_amount"),
            F.avg("speed_mph").alias("avg_speed_mph"),
            F.avg("passenger_count").alias("avg_passenger_count"),
            F.sum("total_amount").alias("total_revenue"),
            F.sum(F.when(F.col("is_airport_pickup"), 1).otherwise(0)).alias("airport_pickup_count"),
            F.sum(F.when(F.col("is_airport_dropoff"), 1).otherwise(0)).alias("airport_dropoff_count")
        )
        .withColumn("airport_pickup_pct", F.col("airport_pickup_count") / F.col("trip_count") * 100)
        .withColumn("airport_dropoff_pct", F.col("airport_dropoff_count") / F.col("trip_count") * 100)
    )


@dlt.table(
    name="gold_daily_time_patterns",
    comment="Daily aggregated taxi metrics by time of day",
    partition_cols=["pickup_date"]
)
def gold_daily_time_patterns():
    """
    Gold layer: Patterns quotidiens par moment de la journée.
    """
    return (
        dlt.read("silver_taxi_features")
        .groupBy(
            "pickup_date",
            "pickup_day_of_week",
            "time_of_day"
        )
        .agg(
            F.count("*").alias("trip_count"),
            F.avg("trip_distance").alias("avg_trip_distance"),
            F.avg("trip_duration_minutes").alias("avg_trip_duration"),
            F.avg("fare_amount").alias("avg_fare_amount"),
            F.avg("tip_amount").alias("avg_tip_amount"),
            F.avg("speed_mph").alias("avg_speed_mph"),
            F.sum("total_amount").alias("total_revenue"),
            F.countDistinct("PULocationID").alias("unique_pickup_locations"),
            F.countDistinct("DOLocationID").alias("unique_dropoff_locations"),
            F.sum(F.when(F.col("payment_type") == 1, 1).otherwise(0)).alias("credit_card_count"),
            F.sum(F.when(F.col("payment_type") == 2, 1).otherwise(0)).alias("cash_count")
        )
        .withColumn("credit_card_pct", F.col("credit_card_count") / F.col("trip_count") * 100)
        .withColumn("cash_pct", F.col("cash_count") / F.col("trip_count") * 100)
    )


@dlt.table(
    name="gold_location_pair_metrics",
    comment="Location pair metrics for route optimization",
    partition_cols=["pickup_date"]
)
def gold_location_pair_metrics():
    """
    Gold layer: Analyse des trajets (Pickup-Dropoff).
    """
    return (
        dlt.read("silver_taxi_features")
        .groupBy(
            "pickup_date",
            "PULocationID",
            "DOLocationID"
        )
        .agg(
            F.count("*").alias("route_trip_count"),
            F.avg("trip_distance").alias("avg_route_distance"),
            F.avg("trip_duration_minutes").alias("avg_route_duration"),
            F.avg("fare_amount").alias("avg_route_fare"),
            F.avg("speed_mph").alias("avg_route_speed"),
            F.min("trip_duration_minutes").alias("min_route_duration"),
            F.max("trip_duration_minutes").alias("max_route_duration")
        )
        .filter("route_trip_count >= 5")
    )