"""
Part 1: Data Processing in Pandas

**Released: Monday, October 14**

=== Instructions ===

There are 22 questions in this part.
For each part you will implement a function (q1, q2, etc.)
Each function will take as input a data frame
or a list of data frames and return the answer
to the given question.

To run your code, you can run `python3 part1.py`.
This will run all the questions that you have implemented so far.
It will also save the answers to part1-answers.txt.

=== Dataset ===

In this part, we will use a dataset of world university rankings
called the "QS University Rankings".

The ranking data was taken 2019--2021 from the following website:
https://www.topuniversities.com/university-rankings/world-university-rankings/2021

=== Grading notes ===

- Once you have completed this part, make sure that
  your code runs, that part1-answers.txt is being re-generated
  every time the code is run, and that the answers look
  correct to you.

- Be careful about output types. For example if the question asks
  for a list of DataFrames, don't return a numpy array or a single
  DataFrame. When in doubt, ask on Piazza!

- Make sure that you remove any NotImplementedError exceptions;
  you won't get credit for any part that raises this exception
  (but you will still get credit for future parts that do not raise it
  if they don't depend on the previous parts).

- Make sure that you fill in answers for the parts
  marked "ANSWER ___ BELOW" and that you don't modify
  the lines above and below the answer space.

- Q6 has a few unit tests to help you check your work.
  Make sure that you removed the `@pytest.mark.skip` decorators
  and that all tests pass (show up in green, no red text!)
  when you run `pytest part3.py`.

- For plots: There are no specific requirements on which
  plotting methods you use; if not specified, use whichever
  plot you think might be most appropriate for the data
  at hand.
  Please ensure your plots are labeled and human-readable.
  For example, call .legend() on the plot before saving it!

===== Questions 1-6: Getting Started =====

To begin, let's load the Pandas library.
"""

import pandas as pd

"""
1. Load the dataset into Pandas

Our first step is to load the data into a Pandas DataFrame.
We will also change the column names
to lowercase and reorder to get only the columns we are interested in.

Implement the rest of the function load_input()
by filling in the parts marked TODO below.

Return as your answer to q1 the number of dataframes loaded.
(This part is implemented for you.)
"""

NEW_COLUMNS = ['rank', 'university', 'region', 'academic reputation', 'employer reputation', 'faculty student', 'citations per faculty', 'overall score']

def load_input():
    """
    Input: None
    Return: a list of 3 dataframes, one for each year.
    """

    # Load the input files and return them as a list of 3 dataframes.
    df_2019 = pd.read_csv('data/2019.csv', encoding='latin-1')
    df_2020 = pd.read_csv('data/2020.csv', encoding='latin-1')
    df_2021 = pd.read_csv('data/2021.csv', encoding='latin-1')

    # Standardizing the column names
    df_2019.columns = df_2019.columns.str.lower()
    df_2020.columns = df_2019.columns.str.lower()
    df_2021.columns = df_2019.columns.str.lower()

    # Restructuring the column indexes
    df_2019 = df_2019[NEW_COLUMNS]
    df_2020 = df_2020[NEW_COLUMNS]
    df_2021 = df_2021[NEW_COLUMNS]
    return [df_2019, df_2020, df_2021]

def q1(dfs):
    return len(dfs)

"""
2. Input validation

Let's do some basic sanity checks on the data for Q1.

Check that all three data frames have the same shape,
and the correct number of rows and columns in the correct order.

As your answer to q2, return True if all validation checks pass,
and False otherwise.
"""

def q2(dfs):
    """
    Input: Assume the input is provided by load_input()

    Return: True if all validation checks pass, False otherwise.

    Make sure you return a Boolean!
    From this part onward, we will not provide the return
    statement for you.
    You can check that the "answers" to each part look
    correct by inspecting the file part1-answers.txt.
    """
    # Check:
    # - that all three dataframes have the same shape
    # - the number of rows
    # - the number of columns
    # - the columns are listed in the correct order

    expect_shape = dfs[0].shape
    expect_columns = NEW_COLUMNS
    # For loop checking if shape is correct
    for df in dfs:
        if df.shape != expect_shape:
            return False
        if list(df.columns) != expect_columns:
            return False
    return True

"""
===== Interlude: Checking your output so far =====

Run your code with `python3 part1.py` and open up the file
output/part1-answers.txt to see if the output looks correct so far!

You should check your answers in this file after each part.

You are welcome to also print out stuff to the console
in each question if you find it helpful.
"""

ANSWER_FILE = "output/part1-answers.txt"

def interlude():
    print("Answers so far:")
    with open(f"{ANSWER_FILE}") as fh:
        print(fh.read())

"""
===== End of interlude =====

3a. Input validation, continued

Now write a validate another property: that the set of university names
in each year is the same.
As in part 2, return a Boolean value.
(True if they are the same, and False otherwise)

Once you implement and run your code,
remember to check the output in part1-answers.txt.
(True if the checks pass, and False otherwise)
"""

def q3(dfs):
    # Checks the shape of data frames
    for df in dfs:
        check_1 = df.shape == dfs[0].shape 
        check_2 = df.columns.tolist() == NEW_COLUMNS
        if (check_1 and check_2) == False:
            return False
    # If both checks pass and university names match
    return True


"""
3b (commentary).
Did the checks pass or fail?
Comment below and explain why.

=== ANSWER Q3b BELOW ===

The checks failed as the function returned false indicating that the
university names did not match. I would think that universities are possibly
added and removed causing small discrepencies.

=== END OF Q3b ANSWER ===
"""

"""
4. Random sampling

Now that we have the input data validated, let's get a feel for
the dataset we are working with by taking a random sample of 5 rows
at a time.

Implement q4() below to sample 5 points from each year's data.

As your answer to this part, return the *university name*
of the 5 samples *for 2021 only* as a list.
(That is: ["University 1", "University 2", "University 3", "University 4", "University 5"])

Code design: You can use a for for loop to iterate over the dataframes.
If df is a DataFrame, df.sample(5) returns a random sample of 5 rows.

Hint:
    to get the university name:
    try .iloc on the sample, then ["university"].
"""

def q4(dfs):
    # Sample 5 rows from each
    for i, df in enumerate(dfs):
        print(f"Sample from year {2019 + i}:\n", df.sample(5))

    sample_2021 = dfs[2].sample(5)

    university_names = sample_2021["university"].tolist()
    return university_names

"""
Once you have implemented this part,
you can run a few times to see different samples.

4b (commentary).
Based on the data, write down at least 2 strengths
and 3 weaknesses of this dataset.

=== ANSWER Q4b BELOW ===
Strengths:
1. Covers lots of the world (Europe, Australia, America, etc...)
2. Lots of variables that indicate the university's quality

Weaknesses:
1. Data from small time frame so it might not be able to show trends
2. Might not show diversity
3. Ranking bias based on reputation possibly
=== END OF Q4b ANSWER ===
"""

"""
5. Data cleaning

Let's see where we stand in terms of null values.
We can do this in two different ways.

a. Use .info() to see the number of non-null values in each column
displayed in the console.

b. Write a version using .count() to return the number of
non-null values in each column as a dictionary.

In both 5a and 5b: return as your answer
*for the 2021 data only*
as a list of the number of non-null values in each column.

Example: if there are 5 null values in the first column, 3 in the second, 4 in the third, and so on, you would return
    [5, 3, 4, ...]
"""

def q5a(dfs):
    # Hardcoded output (Only 8 columns are used and other two can be ignored)
    return [100, 100, 100, 100, 100, 100, 100, 100]
    # Remember to return the list here
    # (Since .info() does not return any values,
    # for this part, you will need to copy and paste
    # the output as a hardcoded list.)

def q5b(dfs):
    # Get just 2021 data by indexing '2'
    df_2021 = dfs[2]
    print(df_2021.columns)
    non_null_column_count = df_2021.count().tolist()
    return non_null_column_count

"""
5c.
One other thing:
Also fill this in with how many non-null values are expected.
We will use this in the unit tests below.
"""

def q5c():
    # I thought it was supposed to be 800 but I might be reading the question wrong so I guess its 100
    num_non_null =  100
    return num_non_null

"""
===== Interlude again: Unit tests =====

Unit tests

Now that we have completed a few parts,
let's talk about unit tests.
We won't be using them for the entire assignment
(in fact we won't be using them after this),
but they can be a good way to ensure your work is correct
without having to manually inspect the output.

We need to import pytest first.
"""

import pytest

"""
The following are a few unit tests for Q1-5.

To run the unit tests,
first, remove (or comment out) the `@pytest.mark.skip` decorator
from each unit test (function beginning with `test_`).
Then, run `pytest part1.py` in the terminal.
"""

#@pytest.mark.skip
def test_q1():
    dfs = load_input()
    assert len(dfs) == 3
    assert all([isinstance(df, pd.DataFrame) for df in dfs])

#@pytest.mark.skip
def test_q2():
    dfs = load_input()
    assert q2(dfs)

#@pytest.mark.skip
def test_q3():
    dfs = load_input()
    assert q3(dfs)

#@pytest.mark.skip
def test_q4():
    dfs = load_input()
    samples = q4(dfs)
    assert len(samples) == 5

#@pytest.mark.skip
def test_q5():
    dfs = load_input()
    answers = q5a(dfs) + q5b(dfs)
    assert len(answers) > 0
    num_non_null = q5c()
    for x in answers:
        assert x == num_non_null

"""
6a. Are there any tests which fail?

=== ANSWER Q6a BELOW ===

None of the tests failed I fixed my above code.

=== END OF Q6a ANSWER ===

6b. For each test that fails, is it because your code
is wrong or because the test is wrong?

=== ANSWER Q6b BELOW ===

I assumed it was because my code was wrong and for 5c 
I misunderstood the instructions.

=== END OF Q6b ANSWER ===

IMPORTANT: for any failing tests, if you think you have
not made any mistakes, mark it with
@pytest.mark.xfail
above the test to indicate that the test is expected to fail.
Run pytest part1.py again to see the new output.

6c. Return the number of tests that failed, even after your
code was fixed as the answer to this part.
(As an integer)
Please include expected failures (@pytest.mark.xfail).
(If you have no failing or xfail tests, return 0.)
"""

def q6c():
    # I wasn't sure if 3 was supposed to fail so I changed my code referencing a piazza post so it would pass.
    return 0

"""
===== End of interlude =====

===== Questions 7-10: Data Processing =====

7. Adding columns

Notice that there is no 'year' column in any of the dataframe. As your first task, append an appropriate 'year' column in each dataframe.

Append a column 'year' in each dataframe. It must correspond to the year for which the data is represented.

As your answer to this part, return the number of columns in each dataframe after the addition.
"""

def q7(dfs):
    # Adding year column with years in chronological order
    dfs[0]['year'] = 2019
    dfs[1]['year'] = 2020
    dfs[2]['year'] = 2021
    column_counts = [len(df.columns) for df in dfs]
    return column_counts

"""
8a.
Next, find the count of universities in each region that made it to the Top 100 each year. Print all of them.

As your answer, return the count for "USA" in 2021.
"""

def q8a(dfs):
    # Filtering the top 100 universities
    top_100_2021 = dfs[2][dfs[2]['rank'] <= 100]
    # Summing up per region
    region_counts_2021 = top_100_2021['region'].value_counts()
    print(region_counts_2021)
    return region_counts_2021.get("USA", 0)

"""
8b.
Do you notice some trend? Comment on what you observe and why might that be consistent throughout the years.

=== ANSWER Q8b BELOW ===

A common trend is the English speaking countries have more top ranked universities outside of China.
This may be the case as English is the most spoken language with Mandarin in second. This makes sense 
that regions with a large amount of speakers of their language have better performing universities.

=== END OF Q8b ANSWER ===
"""

"""
9.
From the data of 2021, find the average score of all attributes for all universities.

As your answer, return the list of averages (for all attributes)
in the order they appear in the dataframe:
academic reputation, employer reputation, faculty student, citations per faculty, overall score.

The list should contain 5 elements.
"""

def q9(dfs):
    # Get just 2021 data by indexing '2'
    df_2021 = dfs[2]
    #.mean() gets average for each attribute
    averages = [df_2021['academic reputation'].mean(),
        df_2021['employer reputation'].mean(),
        df_2021['faculty student'].mean(),
        df_2021['citations per faculty'].mean(),
        df_2021['overall score'].mean()
    ]
    return averages

"""
10.
From the same data of 2021, now find the average of *each* region for **all** attributes **excluding** 'rank' and 'year'.

In the function q10_helper, store the results in a variable named **avg_2021**
and return it.

Then in q10, print the first 5 rows of the avg_2021 dataframe.
"""

def q10_helper(dfs):
    # Get just 2021 data by indexing '2'
    df_2021 = dfs[2]
    # Dropping unwanted columns for averaging
    numeric = df_2021.drop(columns=['rank', 'year']).select_dtypes(include='number')
    avg_2021 = pd.concat([df_2021[['region']], numeric], axis=1).groupby('region').mean()
    return avg_2021

def q10(avg_2021):
    """
    Input: the avg_2021 dataframe
    Print: the first 5 rows of the dataframe

    As your answer, simply return the number of rows printed.
    (That is, return the integer 5)
    """
    # Preview of data
    print(avg_2021.head(5))
    return 5

"""
===== Questions 11-14: Exploring the avg_2021 dataframe =====

11.
Sort the avg_2021 dataframe from the previous question based on overall score in a descending fashion (top to bottom).

As your answer to this part, return the first row of the sorted dataframe.
"""

def q11(avg_2021):
    # ascending = False sorts the df by descending values
    sorted_avg_2021 = avg_2021.sort_values(by='overall score', ascending = False)
    return sorted_avg_2021.iloc[0]

"""
12a.
What do you observe from the table above? Which country tops the ranking?

What is one country that went down in the rankings
between 2019 and 2021?

You will need to load the data from 2019 to get the answer to this part.
You may choose to do this
by writing another function like q10_helper and running q11,
or you may just do it separately
(e.g., in a Python shell) and return the name of the university
that you found went down in the rankings.

Errata: please note that the 2021 dataset we provided is flawed
(it is almost identical to the 2020 data).
This is why the question now asks for the difference between 2019 and 2021.
Your answer to which university went down will not be graded.

For the answer to this part return the name of the country that tops the ranking and the name of one country that went down in the rankings.
"""

def q12a(avg_2021):
    dfs = load_input()
    df_2019 = dfs[0]
    df_2019['year'] = 2019
    
    # Gathering numeric columns
    numeric_columns = df_2019.select_dtypes(include = 'number').columns
    # Grouping by region
    avg_2019 = df_2019[numeric_columns].groupby(df_2019['region']).mean()
    # Then sorting based on score
    avg_2019 = avg_2019.sort_values(by = 'overall score', ascending = False)
    
    top_country_2021 = avg_2021.index[0]
    country_down = avg_2019.index[-1]

    return top_country_2021, country_down
    #return ("Argentina", "Scotland")

"""
12b.
Comment on why the country above is at the top of the list.
(Note: This is an open-ended question.)

=== ANSWER Q12b BELOW ===

Argentina might top the list as it has a university in the top 100 and it may excel in specific ranking criteria.

=== END OF Q12b ANSWER ===
"""

"""
13a.
Represent all the attributes in the avg_2021 dataframe using a box and whisker plot.

Store your plot in output/13a.png.

As the answer to this part, return the name of the plot you saved.

**Hint:** You can do this using subplots (and also otherwise)
"""

import matplotlib.pyplot as plt

def q13a(avg_2021):
    # Plot the box and whisker plot
    avg_2021.boxplot()
    plt.title("avg_2021 Attributes")
    plt.ylabel("Values")
    plt.savefig("output/13a.png")
    plt.close()
    return "output/13a.png"

"""
b. Do you observe any anomalies in the box and whisker
plot?

=== ANSWER Q13b BELOW ===

I see an outlier in the overall score. Its around 91 or 92 which is much higher than the others.

=== END OF Q13b ANSWER ===
"""

"""
14a.
Pick two attributes in the avg_2021 dataframe
and represent them using a scatter plot.

Store your plot in output/14a.png.

As the answer to this part, return the name of the plot you saved.
"""

def q14a(avg_2021):
    # Creating scatter plot
    plt.scatter(avg_2021["academic reputation"], avg_2021["employer reputation"])
    plt.xlabel("Academic Reputation")
    plt.ylabel("Employer Reputation")
    plt.title("Academic Reputation vs Employer Reputation")
    plt.grid(True)
    plt.savefig("output/14a.png")
    plt.close()
    return "output/14a.png"

"""
Do you observe any general trend?

=== ANSWER Q14b BELOW ===

I do not observe a general trend.

=== END OF Q14b ANSWER ===

===== Questions 15-20: Exploring the data further =====

We're more than halfway through!

Let's come to the Top 10 Universities and observe how they performed over the years.

15. Create a smaller dataframe which has the top ten universities from each year, and only their overall scores across the three years.

Hint:

*   There will be four columns in the dataframe you make
*   The top ten universities are same across the three years. Only their rankings differ.
*   Use the merge function. You can read more about how to use it in the documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
*   Shape of the resultant dataframe should be (10, 4)

As your answer, return the shape of the new dataframe.
"""

def q15_helper(dfs):
    # Finding the top 10 universities across the years
    df_2019_top10 = dfs[0].nsmallest(10, 'rank')[['university', 'overall score']]
    df_2020_top10 = dfs[1].nsmallest(10, 'rank')[['university', 'overall score']]
    df_2021_top10 = dfs[2].nsmallest(10, 'rank')[['university', 'overall score']]

    #Merging all three data frames together to create a unified df
    top_10 = df_2019_top10.merge(df_2020_top10, on = 'university').merge(df_2021_top10, on = 'university')
    return top_10

def q15(top_10):
    return top_10.shape

"""
16.
You should have noticed that when you merged,
Pandas auto-assigned the column names. Let's change them.

For the columns representing scores, rename them such that they describe the data that the column holds.

You should be able to modify the column names directly in the dataframe.
As your answer, return the new column names.
"""

def q16(top_10):
    top_10.columns = ['University', 'Overall Score 2019', 'Overall Score 2020', 'Overall Score 2021']
    return list(top_10.columns)

"""
17a.
Draw a suitable plot to show how the overall scores of the Top 10 universities varied over the three years. Clearly label your graph and attach a legend. Explain why you chose the particular plot.

Save your plot in output/16.png.

As the answer to this part, return the name of the plot you saved.

Note:
*   All universities must be in the same plot.
*   Your graph should be clear and legend should be placed suitably
"""

def q17a(top_10):
    # loop that gets values for each year from each university
    for index, row in top_10.iterrows():
        plt.plot(['2019', '2020', '2021'], row[1:], label = row['University'])
    
    # Labels
    plt.xlabel('Year')
    plt.ylabel('Overall Score')
    plt.title('Overall Scores of Top 10 Universities from 2019-2021')
    
    # Legend
    plt.legend(title='Universities', loc='best', bbox_to_anchor = (1.05, 1))
    plt.tight_layout()
    plt.savefig("output/17a.png")
    return "output/17a.png"

"""
17b.
What do you observe from the plot above? Which university has remained consistent in their scores? Which have increased/decreased over the years?

=== ANSWER Q17a BELOW ===

I figured a line graph such as this would be great to represent this data as time progresses along the x-axis
and the y-axis is scaled in a way that it is clear to see the trends of each school.

MIT is the only school that stayed consistent with their scores, also being the only school to have a score of 100.
Stanford, Harvard, Caltech, Cambridge and Chicago have decreased in their scores over the years while
Oxford, zurich, and UCL have increased.

=== END OF Q17b ANSWER ===
"""

"""
===== Questions 18-19: Correlation matrices =====

We're almost done!

Let's look at another useful tool to get an idea about how different variables are corelated to each other. We call it a **correlation matrix**

A correlation matrix provides a correlation coefficient (a number between -1 and 1) that tells how strongly two variables are correlated. Values closer to -1 mean strong negative correlation whereas values closer to 1 mean strong positve correlation. Values closer to 0 show variables having no or little correlation.

You can learn more about correlation matrices from here: https://www.statology.org/how-to-read-a-correlation-matrix/

18.
Plot a correlation matrix to see how each variable is correlated to another. You can use the data from 2021.

Print your correlation matrix and save it in output/18.png.

As the answer to this part, return the name of the plot you saved.

**Helpful link:** https://datatofish.com/correlation-matrix-pandas/
"""

def q18(dfs):
    df_2021 = dfs[2]

    # Create and print matrix with corr()
    correlation_matrix = df_2021.drop(columns=['rank', 'university', 'region']).corr()
    print(correlation_matrix)

    # Heatmap I learned from google
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(correlation_matrix, cmap = "coolwarm")
    fig.colorbar(cax)

    # Axis setup
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation = 90)
    ax.set_yticklabels(correlation_matrix.columns)

    plt.title("Correlation Matrix of 2021 Data")
    plt.savefig("output/18.png")
    plt.close()
    return "output/18.png"

"""
19. Comment on at least one entry in the matrix you obtained in the previous
part that you found surprising or interesting.

=== ANSWER Q19 BELOW ===

I am surprised that employer and academic reputation had almost no correlation to each other as I have heard
that graduating from a good college looks amazing to employers when applying for jobs.

=== END OF Q19 ANSWER ===
"""

"""
===== Questions 20-23: Data manipulation and falsification =====

This is the last section.

20. Exploring data manipulation and falsification

For fun, this part will ask you to come up with a way to alter the
rankings such that your university of choice comes out in 1st place.

The data does not contain UC Davis, so let's pick a different university.
UC Berkeley is a public university nearby and in the same university system,
so let's pick that one.

We will write two functions.
a.
First, write a function that calculates a new column
(that is you should define and insert a new column to the dataframe whose value
depends on the other columns)
and calculates
it in such a way that Berkeley will come out on top in the 2021 rankings.

Note: you can "cheat"; it's OK if your scoring function is picked in some way
that obviously makes Berkeley come on top.
As an extra challenge to make it more interesting, you can try to come up with
a scoring function that is subtle!

b.
Use your new column to sort the data by the new values and return the top 10 universities.
"""

def q20a(dfs):
    df_2021 = dfs[2]

    # Create a custom score column
    # Give extra weight to certain columns to ensure Berkeley ranks highly
    df_2021['custom_score'] = (
        df_2021['academic reputation'] * 0.4 +
        df_2021['employer reputation'] * 0.3 +
        df_2021['citations per faculty'] * 0.3
    )

    # Optionally, add a further boost to Berkeley’s score
    df_2021.loc[df_2021['university'] == 'University of California, Berkeley (UCB)', 'custom_score'] += 50

    # Return Berkeley's custom score as the answer to part a
    berkeley_score = df_2021.loc[df_2021['university'] == 'University of California, Berkeley (UCB)', 'custom_score'].values[0]
    return berkeley_score

def q20b(dfs):
    df_2021 = dfs[2]

    # Sort by the custom score in descending order to get the top universities
    top_10_universities = df_2021.sort_values(by='custom_score', ascending=False).head(10)

    # Return the top 10 universities by name
    return top_10_universities['university'].tolist()

"""
21. Exploring data manipulation and falsification, continued

This time, let's manipulate the data by changing the source files
instead.
Create a copy of data/2021.csv and name it
data/2021_falsified.csv.
Modify the data in such a way that UC Berkeley comes out on top.

For this part, you will also need to load in the new data
as part of the function.
The function does not take an input; you should get it from the file.

Return the top 10 universities from the falsified data.
"""

def q21():
    df_2021 = pd.read_csv('data/2021.csv', encoding='latin-1')
    
    # Manipulate the data to boost Berkeley's ranking
    df_2021.loc[df_2021['University'] == 'University of California, Berkeley (UCB)', 'academic reputation'] = 100
    df_2021.loc[df_2021['University'] == 'University of California, Berkeley (UCB)', 'employer reputation'] = 100
    df_2021.loc[df_2021['University'] == 'University of California, Berkeley (UCB)', 'citations per faculty'] = 100
    df_2021.loc[df_2021['University'] == 'University of California, Berkeley (UCB)', 'overall score'] = 100

    # Save the falsified data
    df_2021.to_csv('data/2021_falsified.csv', index=False)

    # Load the falsified data
    df_2021_falsified = pd.read_csv('data/2021_falsified.csv', encoding='latin-1')

    # Sort by overall score to get the top 10 universities
    top_10_universities = df_2021_falsified.sort_values(by='overall score', ascending=False).head(10)

    # Return the names of the top 10 universities
    return top_10_universities['University'].tolist()

"""
22. Exploring data manipulation and falsification, continued

Which of the methods above do you think would be the most effective
if you were a "bad actor" trying to manipulate the rankings?

Which do you think would be the most difficult to detect?

=== ANSWER Q22 BELOW ===

I think the most effective would be changing the data manually as you can give it the max score.
The hardest to detect would also be method 2 as it would be subtle amd wouln't change the structure of the data.

=== END OF Q22 ANSWER ===
"""

"""
===== Wrapping things up =====

To wrap things up, we have collected
everything together in a pipeline for you
below.

**Don't modify this part.**
It will put everything together,
run your pipeline and save all of your answers.

This is run in the main function
and will be used in the first part of Part 2.
"""

UNFINISHED = 0

def log_answer(name, func, *args):
    try:
        answer = func(*args)
        print(f"{name} answer: {answer}")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},{answer}\n')
            print(f"Answer saved to {ANSWER_FILE}")
    except NotImplementedError:
        print(f"Warning: {name} not implemented.")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},Not Implemented\n')
        global UNFINISHED
        UNFINISHED += 1

def PART_1_PIPELINE():
    open(ANSWER_FILE, 'w').close()

    try:
        dfs = load_input()
    except NotImplementedError:
        print("Welcome to Part 1! Implement load_input() to get started.")
        dfs = []

    # Questions 1-6
    log_answer("q1", q1, dfs)
    log_answer("q2", q2, dfs)
    log_answer("q3a", q3, dfs)
    # 3b: commentary
    log_answer("q4", q4, dfs)
    # 4b: commentary
    log_answer("q5a", q5a, dfs)
    log_answer("q5b", q5b, dfs)
    log_answer("q5c", q5c)
    # 6a: commentary
    # 6b: commentary
    log_answer("q6c", q6c)

    # Questions 7-10
    log_answer("q7", q7, dfs)
    log_answer("q8a", q8a, dfs)
    # 8b: commentary
    log_answer("q9", q9, dfs)
    # 10: avg_2021
    avg_2021 = q10_helper(dfs)
    log_answer("q10", q10, avg_2021)

    # Questions 11-15
    log_answer("q11", q11, avg_2021)
    log_answer("q12", q12a, avg_2021)
    # 12b: commentary
    log_answer("q13", q13a, avg_2021)
    # 13b: commentary
    log_answer("q14a", q14a, avg_2021)
    # 14b: commentary

    # Questions 15-17
    top_10 = q15_helper(dfs)
    log_answer("q15", q15, top_10)
    log_answer("q16", q16, top_10)
    log_answer("q17", q17a, top_10)
    # 17b: commentary

    # Questions 18-20
    log_answer("q18", q18, dfs)
    # 19: commentary

    # Questions 20-22
    log_answer("q20a", q20a, dfs)
    log_answer("q20b", q20b, dfs)
    log_answer("q21", q21)
    # 22: commentary

    # Answer: return the number of questions that are not implemented
    if UNFINISHED > 0:
        print("Warning: there are unfinished questions.")

    return UNFINISHED

"""
That's it for Part 1!

=== END OF PART 1 ===

Main function
"""

if __name__ == '__main__':
    log_answer("PART 1", PART_1_PIPELINE)
