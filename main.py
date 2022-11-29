import csv
import math
import os.path
import re
from datetime import datetime
from typing import List, Dict, Any
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader
import pdfkit

mpl.use('TkAgg')

salary_currency_format = {
    "AZN": "Манаты",
    "BYR": "Белорусские рубли",
    "EUR": "Евро",
    "GEL": "Грузинский лари",
    "KGS": "Киргизский сом",
    "KZT": "Тенге",
    "RUR": "Рубли",
    "UAH": "Гривны",
    "USD": "Доллары",
    "UZS": "Узбекский сум"
}

field_naming = {
    'name': 'Название',
    'description': 'Описание',
    'key_skills': 'Навыки',
    'experience_id': 'Опыт работы',
    'premium': 'Премиум-вакансия',
    'employer_name': 'Компания',
    'salary': "Оклад",
    'salary_from': 'Нижняя граница вилки оклада',
    'salary_to': 'Верхняя граница вилки оклада',
    'salary_gross': 'Оклад указан до вычета налогов',
    'salary_currency': 'Идентификатор валюты оклада',
    'area_name': 'Название региона',
    'published_at': 'Дата публикации вакансии'
}

default_field_names = ["№", "Название", "Описание", "Навыки", "Опыт работы",
                       "Премиум-вакансия", "Компания", "Оклад", "Название региона", "Дата публикации вакансии"]


def entry_compare_wrapper(compare_v):
    def entry_compare(value):
        return compare_v in value

    return entry_compare


class Salary:
    currency_to_rub = {
        "AZN": 35.68,
        "BYR": 23.91,
        "EUR": 59.90,
        "GEL": 21.74,
        "KGS": 0.76,
        "KZT": 0.13,
        "RUR": 1,
        "UAH": 1.64,
        "USD": 60.66,
        "UZS": 0.0055,
    }

    def __init__(self, salary_from: str, salary_to: str, salary_currency: str):
        self.salary_from = math.floor(float(salary_from))
        self.salary_to = math.floor(float(salary_to))
        self.salary_currency = salary_currency

    def get_median_in_rubles(self):
        return math.floor(math.floor((int(self.salary_from) + int(self.salary_to)) / 2) *
                          self.currency_to_rub[self.salary_currency])


class Vacancy:
    def __init__(self, name: str, salary: Salary, area_name: str, published_at: str):
        self.name = name
        self.salary = salary
        self.area_name = area_name
        self.published_at = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%S%z").year


class Filter:
    filter_funcs = {
        "": lambda x: True,
        "Название": entry_compare_wrapper
    }

    formatter_for_filter_funcs = {
        "": lambda x: None,
        "Название": lambda x: x.name
    }

    def __init__(self, query: str):
        self.ok = True
        self.query = query
        self.error = ""
        self.func = self.filter_funcs[""]
        self.formatter = self.formatter_for_filter_funcs[""]

        if query != "":
            self.func = self.filter_funcs["Название"](query)
            self.formatter = self.formatter_for_filter_funcs["Название"]

    def check(self, vacancy: Vacancy):
        return self.func(self.formatter(vacancy))


class Report:
    def __init__(self): pass

    @staticmethod
    def generate_image(vac_name: str, years: List[str], gen_sal_stat: Dict[Any, int], filter_sal_stat: Dict[Any, int],
                       gen_vac_stat: Dict[Any, int], filter_vac_stat: Dict[Any, int],
                       cities_for_vac_stat: List[str], cities_vac_stat: List[int],
                       cities_for_freq_stat: List[str], cities_freq_stat: List[float]):
        plt.rcParams['font.size'] = '8'

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        Report.generate_simple_gist(ax1, "Уровень зарплат по годам", years,
                                    {"label": "средняя з/п", "values": [gen_sal_stat[i] for i in years]},
                                    {"label": f"з/п {vac_name}", "values": [filter_sal_stat[i] for i in years]})
        Report.generate_simple_gist(ax2, "Количество вакансий по годам", years,
                                    {"label": "Количество вакансий", "values": [gen_vac_stat[i] for i in years]},
                                    {"label": f"Количество вакансий \n{vac_name}", "values": [filter_vac_stat[i] for i in years]})
        Report.generate_horizontal_gist(ax3, "Уровень зарплат по городам", cities_for_vac_stat, cities_vac_stat)
        Report.generate_pie_gist(ax4, "Доля вакансий по городам", cities_for_freq_stat + ["Другие"],
                                 cities_freq_stat + [1 - sum(cities_freq_stat)])

        fig.tight_layout()

        plt.savefig('graph.png')

    @staticmethod
    def generate_simple_gist(ax, title: str, labels: List[str], data1: Dict[str, str | List[int]],
                             data2: Dict[str, str | List[int]]):
        x = np.arange(len(labels))
        width = 0.35

        ax.cla()
        rects1 = ax.bar(x - width / 2, data1["values"], width, label=data1["label"])
        rects2 = ax.bar(x + width / 2, data2["values"], width, label=data2["label"])
        ax.grid(axis="y")

        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", labelrotation=90)
        ax.legend()

    @staticmethod
    def generate_horizontal_gist(ax, title: str, labels: List[str], data: List[int]):
        y = np.arange(len(labels))
        width = 0.35

        for i in range(len(labels)):
            labels[i] = "\n".join(labels[i].split(" "))
            labels[i] = "-\n".join(labels[i].split("-"))

        ax.cla()
        rects1 = ax.barh(y, data, width, label=labels)
        ax.grid(axis="x")
        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6, verticalalignment='center')
        ax.invert_yaxis()

    @staticmethod
    def generate_pie_gist(ax, title: str, labels: List[str], data: List[float]):
        ax.pie(data, labels=labels, textprops={"size": 6})
        ax.axis('equal')
        ax.set_title(title)


class Analytics:
    def __init__(self, filter_obj: Filter):
        self.salary_year_level = {}
        self.vacancy_year_level = {}
        self.year_level_keys = []

        self.salary_year_level_with_f = {}
        self.vacancy_year_level_with_f = {}
        self.year_level_keys_with_f = []

        self.salary_area_level = {}
        self.vacancy_area_level = {}
        self.area_keys = []

        self.vacancy_count = 0

        self.filter_obj = filter_obj

    def add_to_analytics(self, vacancy: Vacancy):
        salary = vacancy.salary.get_median_in_rubles()

        self.vacancy_count += 1

        if vacancy.published_at not in self.salary_year_level:
            self.salary_year_level[vacancy.published_at] = 0
            self.vacancy_year_level[vacancy.published_at] = 0
            self.year_level_keys.append(vacancy.published_at)

        self.salary_year_level[vacancy.published_at] += salary
        self.vacancy_year_level[vacancy.published_at] += 1

        if vacancy.area_name not in self.salary_area_level:
            self.salary_area_level[vacancy.area_name] = 0
            self.vacancy_area_level[vacancy.area_name] = 0
            self.area_keys.append(vacancy.area_name)

        self.salary_area_level[vacancy.area_name] += salary
        self.vacancy_area_level[vacancy.area_name] += 1

        if vacancy.published_at not in self.salary_year_level_with_f:
            self.salary_year_level_with_f[vacancy.published_at] = 0
            self.vacancy_year_level_with_f[vacancy.published_at] = 0
            self.year_level_keys_with_f.append(vacancy.published_at)

        if self.filter_obj.check(vacancy):
            self.salary_year_level_with_f[vacancy.published_at] += salary
            self.vacancy_year_level_with_f[vacancy.published_at] += 1

    def get_analytics_data(self):
        percent_of_vacancy = math.floor(self.vacancy_count / 100)

        salary_year_level = {}
        salary_year_level_with_f = {}

        stat_for_years: List[List[str]] = []
        salary_stat_for_area = []
        freq_stat_for_area = []

        self.year_level_keys.sort()

        for i in self.year_level_keys:
            salary_year_level[i] = math.floor(self.salary_year_level[i] / self.vacancy_year_level[i])
            if self.vacancy_year_level_with_f[i] != 0:
                salary_year_level_with_f[i] = math.floor(
                    self.salary_year_level_with_f[i] / self.vacancy_year_level_with_f[i])
            else:
                salary_year_level_with_f[i] = 0

            stat_for_years.append([i, salary_year_level[i], salary_year_level_with_f[i], self.vacancy_year_level[i],
                                   self.vacancy_year_level_with_f[i]])

        for i in self.area_keys:
            if self.vacancy_area_level[i] >= percent_of_vacancy:
                num1 = self.vacancy_area_level[i] / self.vacancy_count
                num2 = num1 * 10000
                num3 = math.floor(num2 + (1 if num2 * 10 % 10 >= 5 else 0)) / 10000

                salary = math.floor(self.salary_area_level[i] / self.vacancy_area_level[i])

                salary_stat_for_area.append([i, salary])
                freq_stat_for_area.append([i, num3])

        salary_stat_for_area.sort(key=lambda x: x[1], reverse=True)
        freq_stat_for_area.sort(key=lambda x: x[1], reverse=True)

        salary_stat_for_area = salary_stat_for_area[:10]
        freq_stat_for_area = freq_stat_for_area[:10]

        return {
            "salary_year_level": salary_year_level,
            "vacancy_year_level": self.vacancy_year_level,
            "salary_year_level_with_f": salary_year_level_with_f,
            "vacancy_year_level_with_f": self.vacancy_year_level_with_f,
            "salary_stat_for_area": salary_stat_for_area,
            "freq_stat_for_area": freq_stat_for_area,
            "stat_for_years": stat_for_years
        }

    def get_tabled_data(self, analytic: Dict[str, Any]):
        stat_for_years_headers = ["Год", "Средняя зарплата", f"Средняя зарплата -<br>{self.filter_obj.query}",
                                  "Количество вакансий", f"Количество вакансий -<br>{self.filter_obj.query}"]
        stat_for_city_headers = ["Город", "Уровень зарплат", "", "Город", "Доля вакансий"]

        stat_for_city: List[List[str]] = []

        for i in range(len(analytic["salary_stat_for_area"])):
            freq_stat = analytic["freq_stat_for_area"][i]
            num1 = freq_stat[1] * 10000
            num2 = math.floor(num1 + (1 if num1 * 10 % 10 >= 5 else 0)) / 100
            num3 = f"{num2}%"

            stat_for_city.append(analytic["salary_stat_for_area"][i] + ["", freq_stat[0], num3])

        return {
            "stat_for_years_headers": stat_for_years_headers,
            "stat_for_city_headers": stat_for_city_headers,
            "stat_for_years": analytic["stat_for_years"],
            "stat_for_city": stat_for_city
        }

    def print_analytics_to_img(self, analytic: Dict[str, Any]):
        Report.generate_image(self.filter_obj.query, self.year_level_keys, analytic["salary_year_level"],
                              analytic["salary_year_level_with_f"], analytic["vacancy_year_level"],
                              analytic["vacancy_year_level_with_f"],
                              list(map(lambda x: x[0], analytic["salary_stat_for_area"])),
                              list(map(lambda x: x[1], analytic["salary_stat_for_area"])),
                              list(map(lambda x: x[0], analytic["freq_stat_for_area"])),
                              list(map(lambda x: x[1], analytic["freq_stat_for_area"])))

    def print_analytics(self, analytic: Dict[str, Any]):
        salary_area_level = "{" + ", ".join(map(lambda x: f"'{x[0]}': {x[1]}", analytic["salary_stat_for_area"])) + "}"
        vacancy_area_level = "{" + ", ".join(map(lambda x: f"'{x[0]}': {x[1]}", analytic["freq_stat_for_area"])) + "}"

        print("Динамика уровня зарплат по годам: " + str(analytic["salary_year_level"]))
        print("Динамика количества вакансий по годам: " + str(analytic["vacancy_year_level"]))
        print("Динамика уровня зарплат по годам для выбранной профессии: " + str(analytic["salary_year_level_with_f"]))
        print("Динамика количества вакансий по годам для выбранной профессии: " + str(analytic["vacancy_year_level_with_f"]))
        print("Уровень зарплат по городам (в порядке убывания): " + salary_area_level)
        print("Доля вакансий по городам (в порядке убывания): " + vacancy_area_level)


class DataSet:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.vacancies_objects = []

        self.ok = True
        self.error = ""

    def parse(self, analytics_obj: Analytics):
        self.vacancies_objects.clear()
        self.ok = True
        self.error = ""

        file = open(self.file_name, mode="r", encoding="utf-8-sig")
        csv_dict_reader = csv.DictReader(file, delimiter=",")
        csv_headers = csv_dict_reader.fieldnames
        empty = True

        if csv_headers is None:
            self.ok = False
            self.error = "Пустой файл"
            return

        for row in csv_dict_reader:
            vacancy_complete = self.check_and_format_row(row, csv_headers)

            if vacancy_complete:
                empty = False
                analytics_obj.add_to_analytics(self.create_vacancy(row))

        if empty:
            self.ok = False
            self.error = "Нет данных"
        elif len(analytics_obj.year_level_keys) == 0:
            self.ok = False
            self.error = "Ничего не найдено"

        file.close()

    @staticmethod
    def check_and_format_row(row: Dict[str, str], headers: List[str]):
        for header in headers:
            if not row[header]:
                return False

            string = re.sub(r'<[^>]*>', "", row[header])
            string_arr = string.split("\n")
            string_arr_len = len(string_arr)
            for i in range(0, string_arr_len):
                string_arr[i] = " ".join(string_arr[i].split())
            row[header] = string_arr[0] if string_arr_len == 1 else string_arr
        return True

    @staticmethod
    def create_vacancy(row: Dict[str, str]):
        salary = Salary(row['salary_from'], row['salary_to'], row['salary_currency'])
        vacancy = Vacancy(row['name'], salary, row["area_name"], row["published_at"])

        return vacancy


class InputConnect:
    def __init__(self):
        self.file_name_query = input("Введите название файла: ")
        self.filter_query = input("Введите название профессии: ")
        self.filter = Filter(self.filter_query)
        self.errors = []
        self.sort_desc = False
        self.check_query()

    def check_query(self):
        self.errors.clear()
        self.sort_desc = False

        if not self.filter.ok:
            self.errors.append(self.filter.error)


input_connect = InputConnect()

if len(input_connect.errors) > 0:
    for error in input_connect.errors:
        print(error)
else:
    dataset = DataSet(input_connect.file_name_query)
    analytics = Analytics(input_connect.filter)
    dataset.parse(analytics)

    if dataset.ok:
        analytic_obj = analytics.get_analytics_data()
        analytics.print_analytics(analytic_obj)
        analytics.print_analytics_to_img(analytic_obj)
        tabled_data = analytics.get_tabled_data(analytic_obj)

        graph = "file://" + os.getcwd() + "/graph.png"

        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template("pdf_template.html")
        pdf_template = template.render({"graph": graph, "vac_name": analytics.filter_obj.query,
                                        "table1Headers": tabled_data["stat_for_years_headers"],
                                        "table1Data": tabled_data["stat_for_years"],
                                        "table2Headers": tabled_data["stat_for_city_headers"],
                                        "table2Data": tabled_data["stat_for_city"]})

        pdfkit.from_string(pdf_template, "report.pdf", options={"enable-local-file-access": ""})

    else:
        print(dataset.error)
