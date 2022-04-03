import sys
import os
import pandas as pd
from tabulate import tabulate

class AnomalyDataPartitioner:
    def __init__(self, params={}):
        self.params = params

        self.contamination_ratio    = params.get('contamination_ratio') or 0.05
        self.normal_count           = params.get('normal_count') or 1000
        self.data_file              = params.get('data_file')
        self.label_normal           = params.get('label_normal') or 1
        self.label_anomaly          = params.get('label_anomaly') or -1
        self.label_column           = params.get('label_column')
        self.output_train_file      = params.get('output_train_file') or 'output.csv'
        self.output_validation_file = params.get('output_validation_file') or 'validation.csv'

        self.anomaly_count = int(self.normal_count * self.contamination_ratio)

    def execute(self):
        df = pd.read_csv(self.data_file)

        df_normal       = df[df[self.label_column] == self.label_normal].sample(n=self.normal_count, random_state=1)
        df_anomalies    = df[df[self.label_column] == self.label_anomaly].sample(n=self.anomaly_count, random_state=1)
        df_validation   = df_normal.append(df_anomalies)
        df_train        = df.drop(df_validation.index)

        print("Saving train set to {}...".format(self.output_train_file))
        df_train.to_csv(self.output_train_file, index=False)

        print("Saving validation set to {}...".format(self.output_validation_file))
        df_validation.to_csv(self.output_validation_file, index=False)

        num_train_normal    = len(df_normal)
        num_train_anomalies = len(df_anomalies)
        num_train           = len(df_train)

        num_validation_normal       = len(df_validation[df_validation[self.label_column] == self.label_normal])
        num_validation_anomalies    = len(df_validation[df_validation[self.label_column] == self.label_anomaly])
        num_validation              = num_validation_normal + num_validation_anomalies
        validation_cont_ratio       = num_validation_anomalies / num_validation

        columns = [
            "# Normal (T)",
            "# Anomalies (T)",
            "Total",
            "Cont. Ratio",
            "# Normal (V)",
            "# Anomalies (V)",
            "Total",
            "Cont. Ratio",
            "T File",
            "V File"
        ]

        table = [
            [
               num_train_normal,
               num_train_anomalies,
               num_train,
               self.contamination_ratio,
               num_validation_normal,
               num_validation_anomalies,
               num_validation,
               validation_cont_ratio,
               self.output_train_file,
               self.output_validation_file
            ]
        ]

        print(tabulate(table, headers=columns, tablefmt="grid"))

        print("Done.")
