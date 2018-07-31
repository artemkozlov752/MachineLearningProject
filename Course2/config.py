import yaml
import os


class Config(object):
    data = yaml.load(open(os.path.join('Config', 'config.yml'), 'r'))

    @staticmethod
    def get_dataset_path(week, dataset_name):
        main_data = Config.data[week][dataset_name]
        return main_data['path']

    @staticmethod
    def get_feature_columns(week, dataset_name):
        main_data = Config.data[week][dataset_name]
        return main_data['columns']['features']

    @staticmethod
    def get_label_column(week, dataset_name):
        main_data = Config.data[week][dataset_name]
        return main_data['columns']['label']

    @staticmethod
    def get_answer_path(week, dataset_name):
        main_data = Config.data[week][dataset_name]
        return main_data['answer']