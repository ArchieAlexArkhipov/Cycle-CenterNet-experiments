# 1 тренировка

## Тренировка CenterNet на WTW-dataset.

124 эпохи

batch size = 2

backbone = ResNet18

optimizer = SGD lr=0.0025, momentum=0.9, weight_decay=0.0001

lr policy="step", warmup="linear", warmup_iters=1000, warmup_ratio=0.001, step=[10, 20, 30, 40, 50]

Остальное можно посмотреть в конфиге.

Лучшие метрики на тестовой выборке на 98 эпохе.
![metrics](imgs/metrics.jpg "metrics")

Лосс с немного обрезанным началом, чтобы было нагляднее.
![loss](imgs/loss.png "loss")

## Анализ ошибок с трешхолдом 0.3

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

# 2 тренировка с параметрами статьи

* test_cfg=dict(topk= ~~100~~ 300 или 2000, local_maximum_kernel= ~~3~~ 1, max_per_img= ~~100~~ 300 или 2000)

* lr=0.00125, step on 90, 120

Лосс с немного обрезанным началом, чтобы было нагляднее.
![loss2](imgs/loss2.png "loss2")

## Анализ ошибок с трешхолдом 0.2

Наклоненные фото с плохой разметкой, множественные пересекающиеся боксы.
![inclined](bad_predictions/70ec95725913a2d9576a8111e6511551e7bc5583_0.021.jpg "inclined")

Таблица без границ, множественные пересекающиеся боксы.
![bad_gt](bad_predictions/a685fd5ae2a72df781f68a4acc808e087954fbfd_0.062.jpg "bad_gt")

Проблема с длинными ячейками и едой.
![food](bad_predictions/mit_google_image_search-10918758-b341ccf7117460c2ab4ed111422e6b8b90d15e41_0.038.jpg "food")
# 5 тренировка resnet34 wandb batch 32

Почему-то не сохранилось лога на train, только на val. 
* test_cfg=dict(topk= ~~100~~ 300 или 2000, local_maximum_kernel= ~~3~~ 1, max_per_img= ~~100~~ 300 или 2000)
* lr=0.00125, step on 90, 120

* resnet34, то есть не будет чекпоинта CenterNet, будет только чекпоинт resnet34 
* wandb 
* batch 32


 Лоссы и метрики https://wandb.ai/centernet/CenterNet/runs/2ae7182x?workspace=user-archiealexarkhipov

## Анализ ошибок с трешхолдом 0.2
Слишком много распознаваний.
![alot](bad_predictions/IMG_0521_0.0.jpg "alot")

Вертикальные и длинные ячейки.
![vertical](bad_predictions/O1CN01nPW7wG1W6VOIVhZ8k_!!6000000002739-0-lxb_0.008.jpg "vertical")

Очень много ложных детекций у еды.
!["alot_food"](bad_predictions/table_spider_00170_0.0.jpg "alot_food")
# Average Precision summary

|                | IoU 0.5:0.95, all area, max 100 dets | IoU 0.5, all area, max 1000 dets | IoU 0.75, all area, max 1000 dets | IoU 0.5:0.95, **small** area, max 1000 dets | IoU 0.5:0.95, medium area, max 1000 dets | IoU 0.5:0.95, large area, max 1000 dets |
|----------------|--------------------------------------|----------------------------------|-----------------------------------|---------------------------------------------|------------------------------------------|-----------------------------------------|
| 1 baseline     | **0.448**                            | 0.511                            | 0.492                         | 0.115                                       | 0.473                                    | **0.651**                               |
| 2 paper params, max 300dets | 0.306                                | 0.492                            | 0.485                             | 0.299                                       | 0.635                              | 0.437                                   |
| 2 paper params, max 2000dets | 0.306 | 0.602 | 0.587 | 0.562 | **0.742** | 0.443 |
| 5 ResNet34, max 300dets    | 0.317                                | 0.527                        | 0.517                             | 0.303                                   | 0.617                                    | 0.43                                    |
| 5 ResNet34, max 2000dets | 0.317 | **0.645** | **0.623** | **0.565** | 0.728 | 0.437 |


# Notes
* max 958 gt bboxes in val dataset.