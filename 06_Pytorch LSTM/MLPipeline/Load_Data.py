
import pandas as pd

class Load_Data:

    def load_data(self):
        """# Reading the Data """
        data = pd.read_csv('Input/review_data (1).csv')
        # Selecting the needed Column
        data = data[['content', 'score']]
        return data
