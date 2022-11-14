## Тренировка CenterNet на WTW-dataset.

124 эпохи

batch size = 2

backbone = ResNet18

optimizer = SGD lr=0.0025, momentum=0.9, weight_decay=0.0001

lr policy="step", warmup="linear", warmup_iters=1000, warmup_ratio=0.001, step=[10, 20, 30, 40, 50]

Остальное можно посмотреть в конфиге.

Лучшие метрики на тестовой выборке на 98 эпохе.
![metrics](metrics.jpg "metrics")

Лосс с немного обрезанным началом, чтобы было нагляднее.
![loss](loss.png "loss")

## Анализ ошибок

Просмотрел топ 100 плохих распознаваний

> Сверху оригинал, снизу предсказание

* Таблицы с большим количеством ячеек, например, Excel. Проблема нежирных контуров.

![excel](bad_predictions/0abd8c28799d176daf5839a227811b035fbf10a3_0.079.jpg "excel")

* В датасете есть наклоненные таблицы со странной разметкой

![inclined](bad_predictions/70ec95725913a2d9576a8111e6511551e7bc5583_0.087.jpg "inclined")

* Перевернутые таблицы, а также таблицы с широкими ячейками

![vertical](bad_predictions/20200211201300234803-0_0.026.jpg "vertical")

* Неправильная изначальная разметка, проблема с широкими ячейками

![mistake](bad_predictions/IMG_0437_0.0.jpg "mistake")

* Изогнутые таблицы на упаковках

![food](bad_predictions/mit_google_image_search-10918758-b817d0b82d29e438aaaae3564949d79628b6f4ee_0.0.jpg "food")