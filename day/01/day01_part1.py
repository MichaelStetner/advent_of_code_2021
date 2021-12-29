import sys

import pyspark
import pyspark.sql.functions as f

# Usual setup
input_filename = '/Users/stetner/adventofcode/2021/day/01/input.txt' #sys.argv[1]
spark = pyspark.sql.SparkSession.builder.getOrCreate()

# Read data, which is one number per line representing the depth of the ocean
input_data = spark.read.csv(input_filename, schema='depth INT')

# My strategy is to enrich each depth with the "previous depth".
# Then, I'll count the rows where depth is greater than previous depth

# Doing the "previous depth" requires some setup. I need to use the lag()
# function, but that requires a window with orderBy().

# I want to keep the data in the original order, so I need to add a column with
# the original sequence
data = input_data.withColumn('seq', f.monotonically_increasing_id())

seq_window = pyspark.sql.Window.orderBy('seq')
data = data.withColumn('prev_depth', f.lag(data['depth']).over(seq_window))

# Finally, solve the problem! Count the number of measurements that are larger
# than the previous measurement
ans = data.filter(data['depth'] > data['prev_depth']).count()
print(ans)