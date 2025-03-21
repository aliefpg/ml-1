from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Augmentation untuk pelatihan (training data)
train_datagen = ImageDataGenerator(
    rescale=1./255,      # Normalisasi gambar
    shear_range=0.2,     # Shear transformation
    zoom_range=0.2,      # Zoom gambar
    horizontal_flip=True # Flip horizontal
)

# Tidak ada augmentasi untuk data uji, hanya normalisasi
test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Memuat data latih dan data uji
train_generator = train_datagen.flow_from_directory(
    'dataset/image/train/',  # Folder data latih
    target_size=(150, 150),    # Ukuran gambar yang diubah
    batch_size=4,
    class_mode='binary'        # Klasifikasi biner
)

test_generator = test_datagen.flow_from_directory(
    'dataset/image/test/',   # Folder data uji
    target_size=(150, 150),
    batch_size=4,
    class_mode='binary'        # Klasifikasi biner
)

valid_generator = valid_datagen.flow_from_directory(
    'dataset/image/valid/',   # Folder data uji
    target_size=(150, 150),
    batch_size=4,
    class_mode='binary'        # Klasifikasi biner
)
