from .util import Dataset
from .dch import DCH

def train(train_img, database_img, query_img, config):
    model = DCH(config)
    img_database = Dataset(database_img, config)
    img_query = Dataset(query_img, config)
    img_train = Dataset(train_img, config)
    model.train(img_train)
    return model.save_file

def validation(database_img, query_img, config):
    model = DCH(config)
    img_database = Dataset(database_img, config)
    img_query = Dataset(query_img, config)
    return model.validation(img_query, img_database, config.R)
