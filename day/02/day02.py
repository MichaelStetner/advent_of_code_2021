import pandas as pd
import pyspark
import pyspark.sql.functions as F

spark = pyspark.sql.SparkSession.builder.getOrCreate()

def load_data(spark, filename):
    input_data = spark.read.csv(
        filename,
        schema='direction STRING, distance INT',
        sep=' '
    )
    return input_data

def part_one(data):
    d = data.groupBy('direction').agg({'distance': 'sum'})
    d = d.select(
            'direction',
            F.when(data['direction'] == 'forward', 'forward').otherwise('depth').alias('axis'),
            F.when(data['direction'] == 'up', -d['sum(distance)']).otherwise(d['sum(distance)']).alias('displacement')
        )
    d = d.groupBy('axis').agg({'displacement': 'sum'})
    d = d.withColumnRenamed('sum(displacement)', 'displ')
    result = d.groupBy().agg(F.product(d.displ))
    return result.collect()[0][0]

if __name__ == '__main__':
    # Usual setup
    input_filename = '/Users/stetner/adventofcode/2021/day/02/input.txt'
    
    data = load_data(spark, input_filename)
    print('--------')
    print('The answer to part one is: {}'.format(part_one(data)))
    print('--------')