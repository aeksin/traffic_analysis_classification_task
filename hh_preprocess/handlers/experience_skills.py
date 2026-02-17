"""Обработчик извлечения ключевых технических навыков через регулярные выражения."""

import logging

import pandas as pd

from ..context import PipelineContext
from .base import Handler

logger = logging.getLogger(__name__)


class SkillFeaturesHandler(Handler):
    """Извлечение навыков с использованием Regex-паттернов (поиск по корням)."""

    SKILL_PATTERNS = {
        "tech_python": r"python|django|flask|fastapi|пайтон",
        "tech_java": r"java\b|spring|hibernate|maven|gradle|джава",
        "tech_js_front": r"javascript|js\b|typescript|react|redux|vue|angular|node|frontend|фронтенд|верстка|html|css|jquery",
        "tech_c_stack": r"c\+\+|cpp|c#|\.net|dotnet|сишарп",
        "tech_php": r"php|laravel|symfony|yii|bitrix|битрикс",
        "tech_mobile": r"android|ios\b|swift|kotlin|flutter|mobile|мобильн.*разработ",
        "tech_sql_db": r"sql|mysql|postgres|oracle|mssql|баз.*данных|бд|субд|t-sql|pl/sql",
        "tech_data_ml": r"pandas|numpy|pytorch|tensorflow|scikit|machine learning|data science|машинн.*обучен|нейросет|анализ.*данных|cv|nlp",
        "tech_devops": r"docker|kubernetes|k8s|ansible|jenkins|gitlab|ci/cd|linux|bash|администрир.*|devops|системн.*администр",
        "tech_1c_prog": r"1с|1c|программист.*1с|разработчик.*1с|язык.*1с",
        "role_qa_manual": r"qa|test case|bug|тестирован|тестировщик|ручн.*тест|баг.*репорт",
        "role_qa_auto": r"selenium|pytest|autotest|автотест|автоматизац.*тестир|loadrunner|jmeter",
        "role_director": r"директор|генеральный|ceo|cto|cfo|head of|руководитель департамент|начальник отдел|заместитель",
        "role_project_manager": r"project manager|product manager|менеджер проект|управлени.*проект|руковод.*проект|pm\b|product owner",
        "role_teamlead": r"team lead|teamlead|тимлид|лид|ведущий разработ|старший разработ|senior|tech lead",
        "soft_management": r"управлени.*команд|постановк.*задач|планирован|budget|бюджет|стратеги|kpi|отчетност.*руковод",
        "role_accountant_chief": r"главный бухгалтер|главбух|зам.*главного бухгалтер",
        "role_accountant": r"бухгалтер|сдача отчетност|первичн.*документ|счет-фактур|акты|ндс|налог|осн|усн",
        "role_finance": r"экономист|финансов.*анализ|бюджетирован|мсфо|рсу|аудит|финансист|казначей",
        "tech_1c_user": r"1с:предприятие|1с:бухгалтерия|зуп|ут|upp|erp|торговля и склад",
        "role_sales_b2b": r"b2b|оптов.*продаж|тендер|закупк|холодн.*звонк|поиск клиент|переговор|коммерческ.*предложен",
        "role_sales_b2c": r"b2c|розничн.*продаж|касс.*|консультирован|продавец|торгов.*представитель|мерчендайзер",
        "role_marketing": r"marketing|smm|seo|контекстн.*реклам|таргет|копирайт|pr|бренд|маркетолог|реклам|продвижен",
        "role_engineer_constr": r"инженер-конструктор|проектировщик|проектирован|чертеж|autocad|компас|revit|bim",
        "role_engineer_prod": r"инженер-технолог|наладчик|производств|оборудован|станк|чпу|электр|монтаж",
        "role_construction": r"прораб|строитель|смет.*|снип|гост|технадзор|участок",
        "role_logistics": r"логист|вэд|таможн|транспорт|перевозк|маршрут",
        "role_warehouse": r"кладовщик|склад|инвентаризац|приемк|отгрузк",
        "role_driver": r"водитель|категори.*b|категори.*c|экспедитор|дальнобойщик",
        "skill_english": r"english|английск|upper|advanced|intermediate|c1|b2|свободный владени",
        "skill_excel_adv": r"vba|сводн.*таблиц|сложн.*формул|excel advanced|продвинутый excel|макрос",
    }

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Извлечь признаки навыков из текстовых полей резюме.

        Объединяет ключевые текстовые колонки (должность, опыт, образование)
        в одну строку, приводит к нижнему регистру и ищет вхождения
        регулярных выражений из SKILL_PATTERNS.

        Аргументы:
            ctx: Контекст пайплайна с загруженным DataFrame.

        Возвращает:
            Обновленный контекст с добавленными бинарными колонками навыков.
        """
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")

        df = ctx.df.copy()

        target_cols = [
            "Ищет работу на должность:",
            "Опыт (двойное нажатие для полной версии)",
            "Последеняя/нынешняя должность",
            "Образование и ВУЗ",
        ]

        existing_cols = [c for c in target_cols if c in df.columns]

        if not existing_cols:
            logger.warning(
                "Не найдены целевые колонки для поиска навыков. Проверьте CSV."
            )
            return ctx

        logger.info(f"Сборка текста из колонок: {existing_cols}")

        full_text = df[existing_cols[0]].fillna("").astype(str)
        for col in existing_cols[1:]:
            full_text = full_text + " " + df[col].fillna("").astype(str)

        full_text = full_text.str.lower()

        logger.info(f"Regex-поиск {len(self.SKILL_PATTERNS)} групп навыков...")

        for feature_name, pattern in self.SKILL_PATTERNS.items():
            df[feature_name] = full_text.str.contains(
                pattern, regex=True, na=False
            ).astype("int8")

        ctx.df = df
        return ctx
