import pyspark
import pyspark.sql.functions as F

def load_data(spark, filename):
    text = spark.read.text(filename)
    text = text.withColumn('row', F.monotonically_increasing_id())
    
    # Convert data to "wide" form. Each entry in the DataFrame corresponds to
    # one character, with its row and column.
    num_cols = text.select(F.first(F.length(text.value))).collect()[0][0]
    data = spark.createDataFrame([], schema='row int, col int, value int')
    # Note that our binary numbers are little endian, 
    # so the rightmost bit is the 0th column
    for i in range(num_cols):
        data = data.union(text.select(
            text.row,
            F.lit(num_cols - i - 1).alias('col'),
            text.value.substr(i + 1, 1)
        ))
    # TODO: There is probably a better way than all these unions.
    # Ideas:
    #   - cross join between text and range of column numbers
    #   - something with F.explode()? https://stackoverflow.com/questions/37864222/transpose-column-to-row-with-spark
    return data

def part_one(data):
    counts = data.groupBy('col').pivot('value').count()
    counts = counts.withColumn('gamma',   F.when(counts['0'] >= counts['1'], 0).otherwise(1))
    counts = counts.withColumn('epsilon', F.when(counts['0'] <= counts['1'], 0).otherwise(1))
    counts = counts.withColumn('gamma_value', counts.gamma * F.pow(2, counts.col))
    counts = counts.withColumn('epsilon_value', counts.epsilon * F.pow(2, counts.col))
    totals = counts.groupBy().sum('gamma_value', 'epsilon_value')
    answer = totals.collect()[0]
    return answer['sum(gamma_value)'] * answer['sum(epsilon_value)']


if __name__ == '__main__':

    # Usual setup
    input_filename = 'input.txt'
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    data = load_data(spark, input_filename)

    print('--------')
    print('The answer to part one is: {}'.format(part_one(data)))
    print('--------')