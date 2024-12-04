""" SEBASTIAN MATTHEWS
    PROF. SPRINT
    CPSC 322
    10/9/24
    PROGRAMMING ASSIGNMENT 2: Classes
"""

import copy
import csv
from tabulate import tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        column = []

        if col_identifier not in self.column_names:
            raise ValueError('col_identifier has an invalid value')

        for row in self.data:
            # If we have a blank column value, skip over it if we're not including missing values.
            if row[self.column_names.index(col_identifier)] == 'NA' and not include_missing_values:
                pass
            else:
                column.append(row[self.column_names.index(col_identifier)])
        return column

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i, row in enumerate(self.data):
            for j, val in enumerate(row):
                try:
                    if val.isnumeric():
                        self.data[i][j] = int(val)
                    else:
                        self.data[i][j] = float(val)
                except:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        displace = 0
        for i in range(len(row_indexes_to_drop)):
            self.data.pop(row_indexes_to_drop[i] - displace)
            displace += 1

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """

        table = []
        header = []
        # 1. Open the file
        infile = open(filename, "r", encoding="utf-8") # Read
        reader = csv.reader(infile)

        # 2. Process the file.
        for i, row in enumerate(reader):
            if i == 0:
                header = row
            else:
                table.append(row)

        self.column_names = header
        self.data = table

        # 3. Close the file
        infile.close()
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w")
        writer = csv.writer(outfile)
        writer.writerow(self.column_names)
        writer.writerows(self.data)
        outfile.close()


    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        res = []
        keys_seen_before = []

        for i, row in enumerate(self.data):
            row_key = self.extract_key_from_key_row(row, key_column_names)
            if row_key not in keys_seen_before:
                keys_seen_before.append(row_key)
            else:
                res.append(i)

        return res

    def extract_key_from_key_row(self, row, key_col_names):
        """Helper function to take a key from a given row.

        Args:
            row(list of elements): the row
            key_col_names(list of str): names of columns to take the keys from
        """
        key_tuple = []

        for key in key_col_names:
            column_idx = self.column_names.index(key)
            key_tuple.append(row[column_idx])

        return str(key_tuple)

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        remove_later = []

        # Make a note of every row we need to remove...
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if str(self.data[i][j]) == '':
                    remove_later.append(i)

        # ...and drop them.
        self.drop_rows(remove_later)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        column = self.get_column(col_name)
        avg = 0
        denom = 0

        for val in column:
            if isinstance(val, (int, float)):
                avg += val
                denom += 1
        avg = avg / denom

        # Replace 'NA' with the average.
        for i in range(len(self.data)):
            if self.data[i][self.column_names.index(col_name)] == 'NA':
                self.data[i][self.column_names.index(col_name)] = avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        stats_table = MyPyTable()
        
        res = []
        col_min = 1000
        col_max = -1000
        mid = 0
        median = 0

        for i in range(len(col_names)):
            column = self.get_column(col_names[i], False)
            if len(column) == 0:
                break
            res.append(col_names[i]) # Attribute

            # Find min and max.
            for val in column:
                if isinstance(val, (int, float)):
                    if val < col_min:
                        col_min = val
                    if val > col_max:
                        col_max = val
                else:
                    column.remove(val)

            # Find median.
            column.sort()
            n = len(column)
            if n % 2 == 0:
                median = (column[n//2 - 1] + column[n//2]) / 2
            else:
                median = column[n//2]

            # Find average and mid.
            avg = sum(column) / len(column)
            mid = (col_min + col_max)/ 2

            res.append(col_min)
            res.append(col_max)
            res.append(mid)
            res.append(avg)
            res.append(median)

            # Reset everything.
            avg = 0
            mid = 0
            col_min = 1000
            col_max = -1000
            stats_table.data.append(res)
            res = []

        return stats_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        third_table = MyPyTable()

        # Create a merged header from the column names in left table and unique to the other table.
        merged_header = self.column_names + [val for val in other_table.column_names if val not in self.column_names]
        third_table.column_names = merged_header

        for row1 in self.data:
            for row2 in other_table.data:
                # If our keys match...
                if self.extract_key_from_key_row(row1, key_column_names) == other_table.extract_key_from_key_row(row2, key_column_names):
                    # ...merge!
                    row3 = row1 + [val for val in row2 if val not in row1]
                    third_table.data.append(row3)

        return third_table

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """

        third_table = MyPyTable()

        # Do everything we did for an inner join...
        merged_header = self.column_names + [val for val in other_table.column_names if val not in self.column_names]
        third_table.column_names = merged_header

        for row1 in self.data:
            key_match = False
            for row2 in other_table.data:
                # If our keys match...
                if self.extract_key_from_key_row(row1, key_column_names) == other_table.extract_key_from_key_row(row2, key_column_names):
                    # ...merge!
                    key_match = True
                    row3 = row1 + [val for val in row2 if val not in row1]
                    third_table.data.append(row3)
            # ... unless we don't see a match!
            if not key_match:
                row3 = row1 + ['NA'] * (len(third_table.column_names) - len(row1))
                third_table.data.append(row3)

        # Get the rest of the values from the other table and put them in the merged table.
        for row2 in other_table.data:
            if not any(self.extract_key_from_key_row(row1, key_column_names) == other_table.extract_key_from_key_row(row2, key_column_names) for row1 in self.data):
                row3 = ['NA'] * len(merged_header)
                for i, key in enumerate(other_table.column_names):
                    row3[third_table.column_names.index(key)] = row2[i]
                third_table.data.append(row3)
        return third_table