import sys

import pyspark
import pyspark.sql.functions as f

def load_data(spark, filename):
    # Read data, which is one number per line representing the depth of the ocean
    input_data = spark.read.csv(input_filename, schema='depth INT')

    # I want to keep the data in the original order, so I need to add a column with
    # the original sequence
    return input_data.withColumn('seq', f.monotonically_increasing_id())

def count_increases(data, col='depth', order_by='seq'):
    # My strategy is to enrich each depth with the "previous depth".
    # Then, I'll count the rows where depth is greater than previous depth
    # Doing the "previous depth" requires some setup. I need to use the lag()
    # function, but that requires a window with orderBy().
    w = pyspark.sql.Window.orderBy(order_by)
    prev_col = 'prev_' + col
    data = data.withColumn(prev_col, f.lag(data[col]).over(w))
    return data.filter(data[col] > data[prev_col]).count()

def part_one(data):
    return count_increases(data)

def part_two(data):
    # For part 2, I need to compute a 3-sample moving average first
    
    # Three sample window (current sample and next two)
    threes = pyspark.sql.Window.orderBy('seq').rowsBetween(pyspark.sql.Window.currentRow, 2)
    
    smoothed = data.select(
        f.min(data['seq']).over(threes).alias('seq'),
        f.sum(data['depth']).over(threes).alias('depth')
    )

    # Remove samples at the end with incomplete windows
    smoothed = smoothed.filter(smoothed['seq'] < (smoothed.count() - 2))

    return count_increases(smoothed)

if __name__ == '__main__':
    # Usual setup
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    input_filename = '/Users/stetner/adventofcode/2021/day/01/input.txt'
    # input_filename = '/Users/stetner/adventofcode/2021/day/01/example.txt'
    # input_filename = sys.argv[1]
    
    data = load_data(spark, input_filename)
    print('--------')
    print('The answer to part one is: {}'.format(part_one(data)))
    print('--------')
    print('The answer to part two is: {}'.format(part_two(data)))
