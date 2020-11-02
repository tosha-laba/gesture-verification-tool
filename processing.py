import os

import matplotlib.pyplot as plt


def crop_image_by_hist(bundle):
    """
    Обрезает исходное изображение по гистограмме.
    """
    image_x = {x[0]: x[1] for x in bundle['image_x'].items() if x[1] >= 8}
    image_y = {x[0]: x[1] for x in bundle['image_y'].items() if x[1] >= 8}

    x0 = min(image_x.keys())
    x1 = max(image_x.keys())
    y0 = min(image_y.keys())
    y1 = max(image_y.keys())

    image_w, image_h = bundle['image'].size

    bundle['image_x_filtered'] = image_x
    bundle['image_y_filtered'] = image_y
    bundle['image_bounds'] = (x0, y0, x1, y1)
    bundle['image_cropped'] = bundle['image'].crop((image_w - x1, image_h - y1, image_w - x0, image_h - y0))


def scale_image_to_ref(bundle):
    """
    Масштабирует обрезанное изображение к эталону.
    """
    img_x, img_y = bundle['image_x_filtered'], bundle['image_y_filtered']
    ref_x, ref_y = bundle['reference_x'], bundle['reference_y']

    img_x_max, img_y_max = max(img_x.values()), max(img_y.values())
    ref_x_max, ref_y_max = max(ref_x.values()), max(ref_y.values())

    bundle['image_x_scaled'] = {k: int(v / img_x_max * ref_x_max) for k, v in img_x.items()}
    bundle['image_y_scaled'] = {k: int(v / img_y_max * ref_y_max) for k, v in img_y.items()}

    img_x_last, img_y_last = max(img_x.keys()), max(img_y.keys())
    ref_x_last, ref_y_last = max(ref_x.keys()), max(ref_y.keys())

    bundle['image_x_scaled'] = {int(k / img_x_last * ref_x_last): v for k, v in bundle['image_x_scaled'].items()}
    bundle['image_y_scaled'] = {int(k / img_y_last * ref_y_last): v for k, v in bundle['image_y_scaled'].items()}


def process_plot(name, caption, ds, legend=None):
    """
    Рисует график и сохраняет его в файл.
    :param name: Имя файла.
    :param caption: Подпись графика.
    :param ds: Набор данных
    :param legend: Легенда (список наименований элементов легенды)
    """
    folder = 'static/images/'

    fig, ax = plt.subplots(nrows=1, ncols=1)
    for i, data in enumerate(ds):
        if legend:
            ax.plot(list(data.keys()), list(data.values()), label=legend[i])
        else:
            ax.plot(list(data.keys()), list(data.values()))
    ax.set_title(caption)
    if legend:
        fig.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    fig.savefig(os.path.join(folder, name))
    plt.close(fig)


def scale_and_crop_fingers(bundle):
    """
    Обрезает изображение до пальцев и сохраняет картинку и график проекции.
    """
    fingers_first = min(bundle['fingers_data'].keys())
    bundle['fingers_data'] = {k - fingers_first: v for k, v in bundle['fingers_data'].items()}

    fingers_last = max(bundle['fingers_data'].keys())
    ref_y_last = max(bundle['reference_y'].keys())

    bundle['fingers_data'] = {int(k / fingers_last * ref_y_last): v for k, v in bundle['fingers_data'].items()}

    process_plot('common_fingers.png', 'Совмещенные проекции на Y', [bundle['fingers_data'], bundle['reference_y']],
                 ['Пальцы       ', 'Эталон'])

    w, h = bundle['image_cropped'].size
    bundle['image_cropped'].crop((0, h / 2, w, h)).save(os.path.join('static/images/', 'image_fingers.png'), 'PNG')


def draw_and_save_plots(bundle):
    """
    Рисует и сохраняет нужные графики.
    """
    process_plot('image_x.png', 'Проекция на X', [bundle['image_x']])
    process_plot('image_y.png', 'Проекция на Y', [bundle['image_y']])
    process_plot('image_cropped_x.png', 'Проекция на X', [bundle['image_x_filtered']])
    process_plot('image_cropped_y.png', 'Проекция на Y', [bundle['image_y_filtered']])
    process_plot('reference_x.png', 'Проекция на X', [bundle['reference_x']])
    process_plot('reference_y.png', 'Проекция на Y', [bundle['reference_y']])
    process_plot('common_x.png', 'Совмещенные проекции на X', [bundle['image_x_scaled'], bundle['reference_x']],
                 ['Изображение       ', 'Эталон'])
    process_plot('common_y.png', 'Совмещенные проекции на Y', [bundle['image_y_scaled'], bundle['reference_y']],
                 ['Изображение       ', 'Эталон'])


def save_pictures(bundle):
    """
    Сохраняет изображения из архива в папку 'static'.
    """
    folder = 'static/images/'

    bundle['image'].save(os.path.join(folder, 'image.png'), 'PNG')
    bundle['image_cropped'].save(os.path.join(folder, 'image_cropped.png'), 'PNG')
    bundle['reference'].save(os.path.join(folder, 'reference.png'), 'PNG')
