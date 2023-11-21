from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

def compute_cosine_similarity(data: DataFrame):
    pair_scores = data \
        .withColumn('xx', F.col('rating1') * F.col('rating1'))\
        .withColumn('yy', F.col('rating2') * F.col('rating2'))\
        .withColumn('xy', F.col('rating1') * F.col('rating2'))
    
    calculated_similarities = pair_scores \
        .groupBy('movie1', 'movie2') \
        .agg(
            F.sum('xy').alias('numerator'),
            (F.sqrt(F.sum('xx')) * F.sqrt(F.sum('yy'))).alias('denominator'),
            F.count('xy').alias('num_pairs')
        )
    
    result = calculated_similarities \
        .withColumn(
            'score',
            F.when(
                F.col('denominator') != 0,
                F.col('numerator') / F.col('denominator')
            ).otherwise(0)
        ).select('movie1', 'movie2', 'score', 'num_pairs')
    
    return result

def get_movie_name(movie_names: DataFrame, movie_id):
    result = movie_names.where(movie_names.movieId == movie_id).select('title').collect()[0]

    return result[0]

# 'local[*]' uses all available cores on the local machine
# good for testing, bad for prod, because we don't want to just use the local machine, but all available in the cluster
spark: SparkSession = SparkSession.builder.appName('movie_recommendations').master('local[*]').getOrCreate()

movie_names = spark.read.csv('./ml-latest-small/movies.csv', header=True)

ratings = spark.read.csv('./ml-latest-small/ratings.csv', header=True).select('userId', 'movieId', 'rating')

average_ratings = ratings.groupBy('movieId').agg(F.round(F.avg('rating'), 2).alias('average_rating')).select('movieId', 'average_rating')

ratings = ratings.join(average_ratings, on='movieId')


ratings = ratings.withColumn(
    'normalized_rating',
    F.round(F.col('rating') - F.col('average_rating'), 2)
)

ratings = ratings.select('userId', 'movieId', 'normalized_rating')

# self join so similarity between each two movies can be performed
# r1.movieId < r2.movieId, because (r1, r2) is the same as (r2, r1)
movie_pairs = ratings.alias('ratings1')\
        .join(ratings.alias('ratings2'),(
            (F.col('ratings1.userId') == F.col('ratings2.userId')) & 
            (F.col('ratings1.movieId') < F.col('ratings2.movieId'))
        ))\
        .select(
            F.col('ratings1.movieId').alias('movie1'),
            F.col('ratings2.movieId').alias('movie2'),
            F.col('ratings1.normalized_rating').alias('rating1'),
            F.col('ratings2.normalized_rating').alias('rating2')
        )

movie_pair_similarities = compute_cosine_similarity(movie_pairs).cache()

score_threshold = 0.95
co_occurence_threshold = 50.

movie_id = 1

filtered_results = movie_pair_similarities.filter(
    ((F.col('movie1') == movie_id) | (F.col('movie2') == movie_id)) &
    (F.col('score') > score_threshold) &
    (F.col('num_pairs') > co_occurence_threshold)
)

results = filtered_results = movie_pair_similarities.sort(F.desc('score')).take(10)

print(f'Top 10 recommendations for {get_movie_name(movie_names, movie_id)}')

for result in results:
    similar_movie_id = result.movie1 if result.movie1 != movie_id else result.movie2

    print(f'{get_movie_name(movie_names, similar_movie_id)}\tscore: {result.score}\tstrength: {result.num_pairs}')

spark.stop()

