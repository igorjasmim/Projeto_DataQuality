import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DataQuality:
    def __init__(self, df: pd.DataFrame) -> None:
        '''
        Iniciate the class with a DataFrame.
        '''
        self.df = df
    
    """
    Check Methods:
    1. Check Nulls
    2. Check Duplicated
    3. Check Numeric Columns
    4. Check Data Types
    5. Check Unique Values
    6. Describe Data
    7. Missing Values Summary
    8. Numeric Columns
    9. Outlier Summary
    10. General Info
    """
    
    def check_nulls(self):
        '''
        Check the number of null values for column.
        '''
        #print("Null values for column:")
        #print(null_values)
        return self.df.isnull().sum()
        
    def check_duplicates(self, remove=False):
        """
        Displays the number of duplicate rows. Provides the option to remove them.
        
        Parameters:
        - remove: If True, remove the duplicate lines.
        """
        duplicate_count = self.df.duplicated().sum()
        # print(f"Duplicated rows: {duplicate_count}")
        
        if remove:
            self.df = self.df.drop_duplicates()
            print("Duplicate rows removed.")
            
        return duplicate_count
    
    def check_numeric_cols(self):
        """
        Check the numeric columns.
        """
        return self.df.select_dtypes(include=np.number).columns
    
    def check_data_types(self):
        """
        Return the types of data from each column.
        """
        return self.df.dtypes
    
    def check_unique_values(self):
        """
        Return the amount of unique values in each column.
        """  
        return self.df.nunique()
    
    def describe_data(self):
        """
        Returns descriptive statistics for numeric columns.
        """
        return self.df.describe()
    
    def missing_values_summary(self):
        """
        Displays a summary of missing values ​​in the DataFrame, including the count and percentage.
        """
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        summary = pd.DataFrame({
            'Missing Values': missing_data,
            'Percentage (%)': missing_percent
        })
        #print(summary[summary['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)) # para imprimir somente os valores diferentes de zero
        return summary
    
    def numeric_cols(self):
        """
        Numeric columns.
        """
        return self.df.select_dtypes(include=np.number)
    
    def outlier_summary(self):
        """
        Identify and display a summary of outliers in numeric columns using IQR.
        """
        Q1 = self.numeric_cols().quantile(0.25)
        Q3 = self.numeric_cols().quantile(0.75)
        IQR = Q3 - Q1

        outliers = ((self.numeric_cols() < (Q1 - 1.5 * IQR)) | (self.numeric_cols() > (Q3 + 1.5 * IQR))).sum()
        print("Summary  of Outliers by Column:")
        print(outliers[outliers > 0])
    
        
    def general_info(self):
        """
        General info.
        """
        print("General info:")
        print(self.df.info())
        


        
    # ----------------------------------------//---------------------------------------- #
    """
    Visualization Methods:
    1. Plot nulls
    2. Plot Data Distribution
    3. Plot BoxPlot
    4. Correlation Matrix
    5. Temporal Analysis
    6. Plot Numeric Distribution
    """
    
    def plot_nulls(self):
        '''
        Generate a bar graph to show the number of null values for column.
        '''
        nulls = self.check_nulls()
        nulls = nulls[nulls > 0]
        if nulls.empty:
            print("There isn't a column with null values.")
            return
        
        plt.figure(figsize=(10,6))
        sns.barplot(x = nulls.index, y = nulls.values)
        plt.title("Null Values for Column")
        plt.ylabel("Number of Null Values")
        plt.xlabel("Columns")
        plt.xticks(rotation=45)
        plt.show()
        
    def plot_data_distribution(self):
        """
        Generate histograms to show the data distribution of numeric columns.
        """
        numeric_cols = self.check_numeric_cols()
        if numeric_cols.empty:
            print("There isn't a numeric column to show.")
            return
        
        self.df[numeric_cols].hist(bins=15, figsize=(12,8), color='skyblue', edgecolor='black')
        plt.suptitle("Distribution of Numeric Data", size=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    def plot_boxplot(self, numeric_cols=None):
        """
        Generate boxplots to detect outliers in the numeric columns.
        The user can select the columns that wish to show. 
        
        Arguments:
        - numeric_cols: list of numeric columns to plot.If None, every numeric column will be show.
        """
        if numeric_cols is None:
            numeric_cols = self.check_numeric_cols()
        else:
            # Check if the input is a columns string separated by comma and do the split:
            if isinstance(numeric_cols, str):
                numeric_cols = [col.strip() for col in numeric_cols.split(',')] # Remove blank spaces
            
            # Checks if selected columns are numeric and exist in the DataFrame
            invalid_columns=[]
            for col in numeric_cols:
                if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col]):
                    invalid_columns.append(col)
            #invalid_columns = [col for col in numeric_cols if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col])]
            if len(invalid_columns) > 0:
                raise ValueError(f"As seguintes colunas não são numéricas ou não existem no DataFrame: {invalid_columns}")
            
        #Check if there are numeric columns to show:
        if len(numeric_cols) == 0:
            print("There isn't a numeric column to show.")
            return
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df[numeric_cols])
        plt.title("Boxplot of Numeric Columns")
        plt.xticks(rotation=45)
        plt.show()
    
    
    def correlation_matrix(self):
        """
        Plots a correlation matrix for the numeric columns.
        """
        # Table of numeric columns:
        numeric_df = self.numeric_cols()
        
        # Check the existence of numeric columns:
        if numeric_df.empty:
            print("There are no numeric columns in the DataFrame to calculate correlation.")
            return
        
        # Plot the Correlation Matrix:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()
    
    def temporal_analysis(self):
        datetime_cols = self.df.select_dtypes(include=['datetime64[ns]'])
        for col in datetime_cols.columns:
            self.df.set_index(col).resample('M').size().plot()
            plt.title(f'Frequency over time in {col}')
            plt.show()
            
    def plot_numeric_dist(self):
        print("Numeric Columns Distribution:")
        numeric_cols = self.numeric_cols()
        for col in numeric_cols.columns:
            plt.figure(figsize=(20, 5))
            sns.histplot(self.df[col], kde=True, bins=30, color='red')
            plt.title(f'Distribution {col}')
            plt.show()
            
    
    # ----------------------------------------//---------------------------------------- #
    """
    General Report
    """
    def general_report(self):
        
        # Check null values:
        self.check_nulls()
        print("\nNull values for column:")
        print(self.check_nulls())
        print(" // ".center(100,"-"))
        print("\n")
        
        # Check duplicated lines:
        print(f"\nDuplicated lines: {self.check_duplicates()}")
        print(" // ".center(100,"-"))
        print("\n")
        
        # Check numeric columns:
        # print("\nNumeric columns:")
        # self.check_numeric_cols()
        # print(" // ".center(100,"-"))
        # print("\n")
        
        # Check Data Types:
        print("\nData Types:")
        print(self.check_data_types())
        print(" // ".center(100,"-"))
        print("\n")
        
        # Check Unique Values:
        print("\nUnique Values:")
        print(self.check_unique_values())
        print(" // ".center(100,"-"))
        print("\n")
        
        # Check Describe Data:
        print("\nDescribe Data:")
        print(self.describe_data())
        print(" // ".center(100,"-"))
        print("\n")
        
        # Missing Values Summary:
        print("\nMissing Values:")
        print(self.missing_values_summary())
        print(" // ".center(100,"-"))
        print("\n")
        
        # Numeric Columns:
        print(self.numeric_cols())
        print(" // ".center(100,"-"))
        print("\n")
        
        # Outlier Summary:
        print(self.outlier_summary())
        print(" // ".center(100,"-"))
        print("\n")
        
        # General Information:
        print(self.general_info())
        print(" // ".center(100,"-"))
        print("\n")
        
        self.plot_nulls()
        print(" // ".center(100,"-"))
        print("\n")
        
        self.plot_data_distribution()
        print(" // ".center(100,"-"))
        print("\n")
        
        self.plot_boxplot()
        print(" // ".center(100,"-"))
        print("\n")
        
        self.correlation_matrix()
        print(" // ".center(100,"-"))
        print("\n")
        
        self.temporal_analysis()
        print(" // ".center(100,"-"))
        print("\n")
        
        self.plot_numeric_dist()
        print(" // ".center(100,"-"))
        print("\n")
        