import json
from zipfile import ZipFile

import yaml
from PIL import Image
from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename

from processing import *
from distances import correlation, chi_square, intersection, bhattacharyya, total_prob, make_decision

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'tmp'
app.secret_key = b'b3BlcmF0aW9uIHk='

APP_NAME = "Идентификация жестов | Антон Завьялов, ПИ-72"


@app.route('/settings', methods=['GET'])
def settings():
    for k, v in request.args.items():
        yaml_config[k] = float(v)

    with open('conf.yaml', 'w') as o:
        yaml.dump(yaml_config, o, default_flow_style=False)

    return render_template('settings.html',
                           app_name=APP_NAME,
                           prob_h=yaml_config['prob_h'],
                           prob_v=yaml_config['prob_v'],
                           prob_v_f=yaml_config['prob_v_f'],
                           prob_f_h=yaml_config['prob_f_h'],
                           prob_f_v=yaml_config['prob_f_v'])


@app.route('/', methods=['POST'])
def process():
    # Upload zip archive
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    f = request.files['file']
    if not f or f.filename == '':
        flash('No selected file')
        return redirect(request.url)

    filename = secure_filename(f.filename)
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Extract content
    with ZipFile(os.path.join(app.config['UPLOAD_FOLDER'], filename)) as archive:
        with archive.open('config.json') as f:
            config = json.load(f)

        bundle = {}
        for k in config:
            v = config[k]
            if k == 'type' or k == 'fingers':
                bundle[k] = v
                continue
            with archive.open(v) as f:
                if v[-3:] == 'png':
                    bundle[k] = Image.open(f).copy()
                else:
                    bundle[k] = {}
                    for x in f.readlines():
                        x = str(x).replace('\\n', '').replace('\\r', '')[2:-1]
                        c, n = map(int, x.split(';'))
                        bundle[k][c] = n

    crop_image_by_hist(bundle)
    scale_image_to_ref(bundle)
    save_pictures(bundle)
    draw_and_save_plots(bundle)

    cor_x, cor_time_x = correlation(bundle['image_x_scaled'], bundle['reference_x'])
    cor_y, cor_time_y = correlation(bundle['image_y_scaled'], bundle['reference_y'])

    chi_x, chi_time_x = chi_square(bundle['image_x_scaled'], bundle['reference_x'])
    chi_y, chi_time_y = chi_square(bundle['image_y_scaled'], bundle['reference_y'])

    int_x, int_time_x = intersection(bundle['image_x_scaled'], bundle['reference_x'])
    int_y, int_time_y = intersection(bundle['image_y_scaled'], bundle['reference_y'])

    bha_x, bha_time_x = bhattacharyya(bundle['image_x_scaled'], bundle['reference_x'])
    bha_y, bha_time_y = bhattacharyya(bundle['image_y_scaled'], bundle['reference_y'])

    wrist = {}
    fingers = {}
    sep = max(bundle['image_y_scaled'].keys()) / 2
    for k, v in bundle['image_y_scaled'].items():
        if k > sep:
            fingers[k] = v
        else:
            wrist[k] = v

    ref_u = {}
    ref_d = {}
    sep = max(bundle['reference_y'].keys()) / 2
    for k, v in bundle['reference_y'].items():
        if k > sep:
            ref_u[k] = v
        else:
            ref_d[k] = v

    wl, fl = ('пальцы', 'кисть') if bundle['fingers'] == 'top' else ('кисть', 'пальцы')

    cor_w, cor_time_w = correlation(wrist, ref_d)
    cor_f, cor_time_f = correlation(fingers, ref_u)

    chi_w, chi_time_w = chi_square(wrist, ref_d)
    chi_f, chi_time_f = chi_square(fingers, ref_u)

    int_w, int_time_w = intersection(wrist, ref_d)
    int_f, int_time_f = intersection(fingers, ref_u)

    bha_w, bha_time_w = bhattacharyya(wrist, ref_d)
    bha_f, bha_time_f = bhattacharyya(fingers, ref_u)

    cor_prob = total_prob(
        [((cor_x + 1) / 2, yaml_config['prob_h']), ((cor_y + 1) / (cor_w + 1), yaml_config['prob_v']),
         ((cor_y + 1) / (cor_f + 1), yaml_config['prob_v_f'])])
    chi_prob = total_prob([(1 - chi_x, yaml_config['prob_h']), ((1 - chi_y) / (1 - chi_w), yaml_config['prob_v']),
                           ((1 - chi_y) / (1 - chi_f), yaml_config['prob_v_f'])])
    int_prob = total_prob([(int_x, yaml_config['prob_h']), (int_y / int_w, yaml_config['prob_v']),
                           (int_y / int_f, yaml_config['prob_v_f'])])
    bha_prob = total_prob(
        [(1 - bha_x, yaml_config['prob_h']), ((1 - bha_y) / (1 - bha_w), yaml_config['prob_v']),
         ((1 - bha_y) / (1 - bha_f), yaml_config['prob_v_f'])])

    bundle['fingers_data'] = fingers if bundle['fingers'] == 'bottom' else wrist
    scale_and_crop_fingers(bundle)

    prob_table = [
        ('Корреляция', cor_prob * 100, make_decision(cor_prob)),
        ('Хи-квадрат', chi_prob * 100, make_decision(chi_prob)),
        ('Пересечение', int_prob * 100, make_decision(int_prob)),
        ('Бхаттачария', bha_prob * 100, make_decision(bha_prob))
    ]

    # Stage 3. Actualization
    fcor, fcor_time = correlation(bundle['fingers_data'], bundle['reference_y'])
    fchi, fchi_time = chi_square(bundle['fingers_data'], bundle['reference_y'])
    fint, fint_time = intersection(bundle['fingers_data'], bundle['reference_y'])
    fbha, fbha_time = bhattacharyya(bundle['fingers_data'], bundle['reference_y'])

    fup = {}
    fdown = {}
    sep = max(bundle['fingers_data'].keys()) / 2
    for k, v in bundle['fingers_data'].items():
        if k > sep:
            fup[k] = v
        else:
            fdown[k] = v

    fcor_w, fcor_time_w = correlation(fup, ref_d)
    fcor_f, fcor_time_f = correlation(fdown, ref_u)

    fchi_w, fchi_time_w = chi_square(fup, ref_d)
    fchi_f, fchi_time_f = chi_square(fdown, ref_u)

    fint_w, fint_time_w = intersection(fup, ref_d)
    fint_f, fint_time_f = intersection(fdown, ref_u)

    fbha_w, fbha_time_w = bhattacharyya(fup, ref_d)
    fbha_f, fbha_time_f = bhattacharyya(fdown, ref_u)

    fcor_prob = total_prob(
        [((cor_x + 1) / 2, yaml_config['prob_f_h']), ((fcor + 1) / (fcor_w + 1), yaml_config['prob_f_v']),
         ((fcor + 1) / (fcor_f + 1), yaml_config['prob_f_v'])])
    fcor_prob = 1 if fcor_prob > 1 else 0 if fcor_prob < 0 else fcor_prob
    fchi_prob = total_prob([(1 - chi_x, yaml_config['prob_f_h']), ((1 - fchi) / (1 - fchi_w), yaml_config['prob_f_v']),
                            ((1 - fchi) / (1 - fchi_f), yaml_config['prob_f_v'])])
    fchi_prob = 1 if fchi_prob > 1 else 0 if fchi_prob < 0 else fchi_prob
    fint_prob = total_prob([(int_x, yaml_config['prob_f_h']), (fint / fint_w, yaml_config['prob_f_v']),
                            (fint / fint_f, yaml_config['prob_f_v'])])
    fint_prob = 1 if fint_prob > 1 else 0 if fint_prob < 0 else fint_prob
    fbha_prob = total_prob([(1 - bha_x, yaml_config['prob_f_h']), ((1 - fbha) / (1 - fbha_w), yaml_config['prob_f_v']),
                            ((1 - fbha) / (1 - fbha_f), yaml_config['prob_f_v'])])
    fbha_prob = 1 if fint_prob > 1 else 0 if fint_prob < 0 else fint_prob

    return render_template('report.html',
                           app_name=APP_NAME,
                           type=config['type'],
                           image_path=config['image'],
                           image_x_path=config['image_x'],
                           image_y_path=config['image_y'],
                           image_x_len=len(bundle['image_x']),
                           image_y_len=len(bundle['image_y']),
                           reference_path=config['reference'],
                           reference_x_path=config['reference_x'],
                           reference_y_path=config['reference_y'],
                           reference_x_len=len(bundle['reference_x']),
                           reference_y_len=len(bundle['reference_y']),
                           calc_table=[('Корреляция (гориз.)',
                                        cor_x,
                                        cor_time_x),
                                       ('Корреляция (верт.)',
                                        cor_y,
                                        cor_time_y),
                                       ('Хи-квадрат (гориз.)',
                                        chi_x,
                                        chi_time_x),
                                       ('Хи-квадрат (верт.)',
                                        chi_y,
                                        chi_time_y),
                                       ('Пересечение (гориз.)',
                                        int_x,
                                        int_time_x),
                                       ('Пересечение (верт.)',
                                        int_y,
                                        int_time_y),
                                       ('Бхаттачария (гориз.)',
                                        bha_x,
                                        bha_time_x),
                                       ('Бхаттачария (верт.)',
                                        bha_y,
                                        bha_time_y)],
                           calc_table_sep=[('Корреляция ({})'.format(wl),
                                            cor_w,
                                            cor_time_w),
                                           ('Корреляция ({})'.format(fl),
                                            cor_f,
                                            cor_time_f),
                                           ('Хи-квадрат ({})'.format(wl),
                                            chi_w,
                                            chi_time_w),
                                           ('Хи-квадрат ({})'.format(fl),
                                            chi_f,
                                            chi_time_f),
                                           ('Пересечение ({})'.format(wl),
                                            int_w,
                                            int_time_w),
                                           ('Пересечение ({})'.format(fl),
                                            int_f,
                                            int_time_f),
                                           ('Бхаттачария ({})'.format(wl),
                                            bha_w,
                                            bha_time_w),
                                           ('Бхаттачария ({})'.format(fl),
                                            bha_f,
                                            bha_time_f)],
                           prob_table=prob_table,
                           fingers_table=[
                               ('Корреляция',
                                fcor,
                                fcor_time),
                               ('Хи-квадрат',
                                fchi,
                                fchi_time),
                               ('Пересечение',
                                fint,
                                fint_time),
                               ('Бхаттачария',
                                fbha,
                                fbha_time)
                           ],
                           fingers_table_sep=[
                               ('Корреляция (верхняя часть)',
                                fcor_f,
                                fcor_time_f),
                               ('Корреляция (нижняя часть)',
                                fcor_w,
                                fcor_time_w),
                               ('Хи-квадрат (верхняя часть)',
                                fchi_time_f,
                                fchi_time_f),
                               ('Хи-квадрат (нижняя часть)',
                                fchi_time_w,
                                fchi_time_w),
                               ('Пересечение (верхняя часть)',
                                fint_f,
                                fint_time_f),
                               ('Пересечение (нижняя часть)',
                                fint_w,
                                fint_time_w),
                               ('Бхаттачария (верхняя часть)',
                                fbha_f,
                                fbha_time_f),
                               ('Бхаттачария (нижняя часть)',
                                fbha_w,
                                fbha_time_w)
                           ],
                           fprob_table=list(map(lambda v: v[1],
                                                filter(lambda i: prob_table[i[0]] == 'Не получен чёткий ответ',
                                                       enumerate([
                                                           ('Корреляция', fcor_prob * 100,
                                                            make_decision(fcor_prob)),
                                                           ('Хи-квадрат', fchi_prob * 100,
                                                            make_decision(fchi_prob)),
                                                           ('Пересечение', fint_prob * 100,
                                                            make_decision(fint_prob)),
                                                           ('Бхаттачария', fbha_prob * 100,
                                                            make_decision(fbha_prob))
                                                       ])))))


@app.route('/', methods=['GET'])
def index():
    return render_template('load.html', app_name=APP_NAME)


if __name__ == 'app':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.mkdir(app.config['UPLOAD_FOLDER'])

    with open('conf.yaml', 'r') as s:
        yaml_config = yaml.safe_load(s)

if __name__ == '__main__':
    app.run()
