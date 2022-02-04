import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window

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

def cumsum(df, result_column, column_to_sum):
    win = Window().rowsBetween(Window.unboundedPreceding, Window.currentRow)
    return df.withColumn(result_column, F.sum(column_to_sum).over(win))

def part_two(data):
    # Calculate delta_aim, take cumsum to get aim
    data = data.withColumn('delta_aim',
                           F.when(data['direction'] == 'up', -data['distance'])
                            .when(data['direction'] == 'down', data['distance'])
                            .otherwise(0))
    data = cumsum(data, 'aim', 'delta_aim')

    # Calculate deltas from forward movement, take cumsum to get
    data = data.withColumn('delta_horizontal',
        F.when(data['direction'] == 'forward', data['distance'])
         .otherwise(0))
    data = data.withColumn('delta_depth',
        F.when(data['direction'] == 'forward', data['distance'] * data['aim'])
         .otherwise(0))
    data = cumsum(data, 'horizontal', 'delta_horizontal')
    data = cumsum(data, 'depth', 'delta_depth')

    # Get final position
    final_position = (data
        .select(
            F.last('horizontal').alias('horizontal'),
            F.last('depth').alias('depth')
        )
        .collect()[0]
    )

    return final_position['horizontal'] * final_position['depth']

if __name__ == '__main__':
    # Usual setup
    input_filename = '/Users/stetner/adventofcode/2021/day/02/input.txt'
    
    data = load_data(spark, input_filename)
    print('--------')
    print('The answer to part one is: {}'.format(part_one(data)))
    print('--------')
    print('The answer to part two is: {}'.format(part_two(data)))
    print('--------')