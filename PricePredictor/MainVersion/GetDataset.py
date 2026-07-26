import kagglehub

# Скачиваем датасет
path = kagglehub.dataset_download("lokeshparab/amazon-products-dataset")

print("Path to dataset files:", path)

# После скачивания датасета переместите Amazon-Products.csv в папку Dataset или поменяйте путь в переменной datasetpath
datasetpath = 'Dataset/Amazon-Products.csv'