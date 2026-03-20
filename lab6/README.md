# Лабораторная работа №6
# Обучение модели семантической сегментации на трехмерных томографических данных


Лабораторная работа основана на примере из документации к библиотеке [TorchIO](https://github.com/TorchIO-project/torchio/blob/main/tutorials/README.md).

Вам необходимо обучить модель сегментации органов на томографических данных.

[Ссылка на блокнот лабораторной работы](https://www.kaggle.com/code/kvsbmstu/torchio-monai-lightning)


В туториале представлен код обучения модели 3D-UNet для сегментации гиппокампа.
Подготовка датасета, добавление специализированных аугментаций для 3D-данных осуществляется с помощью библиотеки [TorchIO](https://torchio.readthedocs.io/).
Более подробное руководство о возможностях добавления аугментаций представлено в еще одном [блокноте](https://colab.research.google.com/github/TorchIO-project/torchio-notebooks/blob/main/notebooks/Data_preprocessing_and_augmentation_using_TorchIO_a_tutorial.ipynb).

Кроме того, поскольку работа с 3D-данными требует гораздо большего объема доступной видеопамяти, разработчики TorchIO реализовали возможность работы с [патчами](https://torchio.readthedocs.io/patches/index.html).


В качестве модели в туториале применена 3D-UNet, которая импортируется из библиотеки [MONAI](https://docs.monai.io/en/latest/).
[MONAI](https://monai.io/) - большой проект по применению методов ИИ в медицине, особый акцент сделан на обработке 3D-медицинских данных.
Репозиторий с туториалами можно найти [здесь](https://github.com/Project-MONAI/tutorials).

Особенностью модели 3D-UNet по сравнению с обычным UNet является применение 3D-сверток.
Различия 2D и 3D сверток хорошо объяснены [здесь](https://learnopencv.com/3d-u-net-brats/), исходный код поста лежит [здесь](https://github.com/spmallick/learnopencv/tree/master/Training_3D_U-Net_Brain_Tumor_Seg).
Автор самостоятельно реализует модель 3D-Unet на PyTorch и обучает ее на данных соревнования по сегментации опухолей мозга [BraTS 2023 Challenge](https://www.synapse.org/Synapse:syn51156910/wiki/621282).

В блокноте лабораторной работы применена библиотека [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), которая является высокоуровневой надстройкой над PyTorch.
Lightning позволяет несколько иначе переорганизовать код, при этом не меняя нейросетевое и алгоритмическое наполнение вашего кода.
Понятные иллюстрации приведены [на главной странице репозитория](https://github.com/Lightning-AI/pytorch-lightning).
С Lightning более удобно настроить правила сохранения чекпоинтов модели, распараллелить вычисления на несколько GPU, задать правила логирования.
Полезно знать о таком решении и попробовать поработать с ним.

Недавно `pytorch_lightning` был переименован в просто `lightning`. Сейчас по обратной совместимости работает такой и такой вариант.

Старый вариант
```
pip install pytorch-lightning
import pytorch_lightning as pl
```
Новый вариант
```
pip install lightning
import lightning as L
```


## Задание

В блокноте лабораторной работы обучается модель сегментации гиппокампа.
Ваша задача - обучить модель на другом датасете, который также доступен через [Medical Segmentation Decathlon](http://medicaldecathlon.com/).

[Ссылка на google drive](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)

Нужно сменить название задачи `task` и `google_id` файла, который можно узнать, если нажать `Поделиться`, `Копировать ссылку`:

`https://drive.google.com/file/d/1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C/view?usp=drive_link`

```
data = MedicalDecathlonDataModule(
    task='Task04_Hippocampus',
    google_id='1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C',
    batch_size=16,
    train_val_ratio=0.8,
)
```
Если будет ошибка, связанная с превышением лимита загрузки файла с google drive, то скопируйте файл на свой google диск и передайте google_id файла своей копии.



При этом в модели `unet` нужно поменять число выходов `out_channels` на число сегментируемых классов в вашей задаче:
```
unet = monai.networks.nets.UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3, # фон и 2 класса
    channels=(8, 16, 32, 64),
    strides=(2, 2, 2),
)
```

Если возникает проблема с переполнением памяти GPU, уменьшите размер батча `batch_size`.
```
data = MedicalDecathlonDataModule(
    task='...',
    google_id='...',
    batch_size=4,
    train_val_ratio=0.8,
)
```


Попробуйте применить не Unet, любую другую архитектуру из MONAI, которая работает с аналогичными 3D данными.


Также нужно попробовать сделать анимацию в формате gif из серии срезов для тестового примера, где на каждом срезе будут наложены предсказываемые маски.








