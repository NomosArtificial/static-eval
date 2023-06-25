from .barbri_task import BarbriTask
from .mpre_task import MPRETask
from .overruling import OverrulingTask
from .tos import ToSTask
from .casehold import CaseHOLDTask
from .legalbench import LBTask

def get_tasks():
    return {
        "barbri": BarbriTask,
        "mpre": MPRETask,
        "overruling": OverrulingTask,
        "tos": ToSTask,
        "casehold": CaseHOLDTask,
        "legalbench": LBTask
    }