PATH_TO_IMAGES_DIR = '/content/gdrive/MyDrive/BE Project/IU-XRAY/frontal/'
PATH_TO_TRAIN_FILE = '/content/gdrive/MyDrive/BE Project/IU-XRAY/train_balanced.csv'
PATH_TO_VAL_FILE = '/content/gdrive/MyDrive/BE Project/IU-XRAY/val_balanced.csv'
PATH_TO_TEST_FILE = '/content/gdrive/MyDrive/BE Project/IU-XRAY/test.csv'

PRE_TRAINED_WEIGHTS_FOR_HEAT_MAP = './model0122.pth.tar'



CLASS_NAMES = ['Atelectasis', 'Calcified Granuloma', 'Calsinosis', 'Cardiomegaly', 'Consolidation', 'Edema',
               'Effusion', 'Emphysema', 'Granulomatous Disease', 'Hernia',
               'Infiltrate', 'Lung', 'Medical Device', 'Nodule', 'Normal', 'Opacity', 'Pneumonia', 'Pneumothorax', 'Spine']

LIST_NN_ARCHITECTURE = ['feature_extractor', 'DENSE-NET-121', 'DENSE-NET-169', 'DENSE-NET-201']
NN_ARCHITECTURE = LIST_NN_ARCHITECTURE[0]
NN_IS_PRE_TRAINED = True

NUM_CLASSES = 19
TRANS_RESIZE = 256
TRANS_CROP = 224

TRAIN_DICT = {
    'Batch Size': 32,
    'Max Epoch': 1000,
    'Learning Rate': 0.001
}

TEST_DICT = {
    'Batch Size': 32,
}

