"""Сборка и конфигурация цепочки обработчиков (Pipeline)."""

from .handlers.base import Handler
from .handlers.bert_embedding import BertEmbeddingHandler
from .handlers.clean_cols import CleanColumnsHandler
from .handlers.clear_control_chars import CleanControlCharsHandler
from .handlers.deduplicate import DeduplicateHandler
from .handlers.enrich_features import EnrichFeaturesHandler
from .handlers.experience_skills import SkillFeaturesHandler
from .handlers.filter_salary_outliers import FilterSalaryOutliersHandler
from .handlers.finalize_arrays import FinalizeArraysHandler
from .handlers.group_city import CityGroupingHandler
from .handlers.job_category import JobCategoryHandler
from .handlers.load_csv import LoadCsvHandler
from .handlers.one_hot import OneHotEncodeHandler
from .handlers.parse_auto import ParseAutoHandler
from .handlers.parse_demographics import ParseGenderAgeHandler
from .handlers.parse_education import ParseEducationHandler
from .handlers.parse_employment_schedule import ParseEmploymentScheduleHandler
from .handlers.parse_experience import ParseExperienceHandler
from .handlers.parse_location import ParseLocationHandler
from .handlers.parse_salary import ParseSalaryHandler
from .handlers.parse_update import ParseResumeUpdateHandler
from .handlers.save_npy import SaveNpyHandler
from .handlers.standardize_numeric import StandardizeNumericHandler


def build_pipeline() -> Handler:
    """Собрать пайплайн обработки в виде цепочки ответственности.

    Возвращает:
        Голова цепочки, с которой нужно начинать выполнение пайплайна.
    """
    head: Handler = LoadCsvHandler()
    cur = head
    for h in [
        CleanControlCharsHandler(),
        CleanColumnsHandler(),
        DeduplicateHandler(),
        ParseGenderAgeHandler(),
        ParseSalaryHandler(),
        FilterSalaryOutliersHandler(),
        ParseLocationHandler(),
        CityGroupingHandler(),
        ParseEmploymentScheduleHandler(),
        ParseExperienceHandler(),
        ParseEducationHandler(),
        ParseResumeUpdateHandler(),
        ParseAutoHandler(),
        JobCategoryHandler(),
        SkillFeaturesHandler(),
        BertEmbeddingHandler(vector_size=25),
        EnrichFeaturesHandler(),
        StandardizeNumericHandler(),
        OneHotEncodeHandler(),
        FinalizeArraysHandler(),
        SaveNpyHandler(),
    ]:
        cur.set_next(h)
        cur = h
    return head
